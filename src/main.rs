use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use battlefield_sim::algorithm::algorithm_by_name;
use battlefield_sim::algorithms::TransformerPolicy;
use battlefield_sim::config::SimConfig;
use battlefield_sim::experiment::{run_session, simulate_round, ExperimentRunner, RunSpec};
use battlefield_sim::persist::{RunError, SqliteStore};
use battlefield_sim::plot::{write_compare_plots, write_metrics_csv};
use battlefield_sim::CloserThanFriend;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "battlefield-sim", about = "Circular battlefield FPV sweep simulator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run one or more experiment rounds and persist results to SQLite.
    Run {
        #[arg(long, default_value_t = 1)]
        rounds: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "nearest_in_range")]
        algo: String,
        #[arg(long, default_value = "./battlefield-sim.sqlite")]
        db: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        log_every_ticks: Option<u64>,
    },
    /// Train the transformer (200 rounds) and evaluate vs closer_than_friend (30 rounds).
    Experiment {
        #[arg(long, default_value_t = 200)]
        train_rounds: u32,
        #[arg(long, default_value_t = 30)]
        eval_rounds: u32,
        #[arg(long, default_value_t = 20260823)]
        seed: u64,
        #[arg(long, default_value = "./experiment.sqlite")]
        db: PathBuf,
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
        #[arg(long, default_value = "models/transformer.json")]
        model: PathBuf,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "battlefield_sim=warn".into()),
        )
        .init();

    let result = match Cli::parse().command {
        Commands::Run {
            rounds,
            seed,
            algo,
            db,
            config,
            log_every_ticks,
        } => run(rounds, seed, &algo, db, config, log_every_ticks),
        Commands::Experiment {
            train_rounds,
            eval_rounds,
            seed,
            db,
            out_dir,
            model,
            lr,
        } => experiment(train_rounds, eval_rounds, seed, db, out_dir, model, lr),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            match e {
                RunError::Config(_) => ExitCode::from(2),
                RunError::Db(_) => ExitCode::from(3),
                RunError::Invariant(_) => ExitCode::from(1),
            }
        }
    }
}

fn run(
    rounds: u32,
    seed: u64,
    algo_name: &str,
    db: PathBuf,
    config_path: Option<PathBuf>,
    log_every_ticks: Option<u64>,
) -> Result<(), RunError> {
    let mut cfg = if let Some(path) = config_path {
        let text = std::fs::read_to_string(&path).map_err(|e| {
            RunError::Invariant(format!("read config {}: {e}", path.display()))
        })?;
        serde_json::from_str::<SimConfig>(&text).map_err(config_from_serde)?
    } else {
        SimConfig::default()
    };
    cfg = cfg.validated()?;
    let algo = algorithm_by_name(algo_name)?;
    let store = SqliteStore::open(&db)?;
    let enemies_total = cfg.enemy_count;
    let mut runner = ExperimentRunner::new(store, algo, cfg, log_every_ticks)?;
    let metrics = runner.run(rounds, seed)?;
    for m in &metrics {
        RunSpec::print_summary(m, algo_name, &db, enemies_total);
    }
    Ok(())
}

fn experiment(
    train_rounds: u32,
    eval_rounds: u32,
    seed: u64,
    db: PathBuf,
    out_dir: PathBuf,
    model: PathBuf,
    lr: f32,
) -> Result<(), RunError> {
    let cfg = SimConfig::default().validated()?;
    println!(
        "experiment seed={seed} train={train_rounds} eval={eval_rounds} lr={lr} model={}",
        model.display()
    );

    let mut policy = TransformerPolicy::new(seed);
    policy.set_name("transformer_train");
    policy.set_greedy(false);
    println!("synthetic pretrain 8000 CE steps on closer_than_friend labels...");
    policy.synthetic_pretrain(8000, lr.max(0.003), seed);
    println!("pretrain done");

    let train_t0 = Instant::now();
    let mut train_rows = Vec::new();
    for i in 0..train_rounds {
        policy.begin_episode();
        // 200 轮仿真上的行为克隆：教师是研究算法 closer_than_friend
        //（用 8 目标距离 + 8 友机距离）。全程 mix=1，避免 REINFORCE 崩到全部弃权。
        policy.set_teacher_mix(1.0);
        let round_seed = seed.wrapping_add(i as u64);
        let metrics = simulate_round(&policy, &cfg, i, round_seed)?;
        policy.finish_episode(&metrics, lr);
        if i == 0 || (i + 1) % 10 == 0 || i + 1 == train_rounds {
            println!(
                "train round {:>3}/{} seed={} killed={} mean_survival={:.4} mean_ops={:.1} decisions={}",
                i + 1,
                train_rounds,
                round_seed,
                metrics.enemies_neutralized,
                metrics.mean_survival_s,
                metrics.mean_compute_ops,
                metrics.decision_count
            );
        }
        train_rows.push(metrics);
    }
    policy.end_training();
    policy
        .save(&model)
        .map_err(|e| RunError::Invariant(format!("save model: {e}")))?;
    println!(
        "training finished in {:.1}s, weights -> {}",
        train_t0.elapsed().as_secs_f64(),
        model.display()
    );

    policy.set_name("transformer");
    policy.set_greedy(true);

    let mut store = SqliteStore::open(&db)?;
    let eval_seed = seed.wrapping_add(10_000);
    println!("eval transformer greedy, {eval_rounds} rounds, base_seed={eval_seed}");
    let t_eval = run_session(
        &mut store,
        &policy,
        &cfg,
        eval_rounds,
        eval_seed,
        None,
    )?;
    println!("eval closer_than_friend, {eval_rounds} rounds, base_seed={eval_seed}");
    let r_eval = run_session(
        &mut store,
        &CloserThanFriend,
        &cfg,
        eval_rounds,
        eval_seed,
        None,
    )?;

    write_metrics_csv(&out_dir.join("results/train_transformer.csv"), "transformer_train", &train_rows)?;
    write_metrics_csv(&out_dir.join("results/eval_transformer.csv"), "transformer", &t_eval)?;
    write_metrics_csv(
        &out_dir.join("results/eval_closer_than_friend.csv"),
        "closer_than_friend",
        &r_eval,
    )?;
    write_compare_plots(&t_eval, &r_eval, &out_dir)?;

    fn mean(xs: impl Iterator<Item = f64>, n: usize) -> f64 {
        xs.sum::<f64>() / n.max(1) as f64
    }
    let n = eval_rounds as usize;
    println!("\n=== eval summary ({} rounds, shared seeds) ===", eval_rounds);
    println!(
        "transformer         killed_mean={:.2}  survival_mean={:.4}  ops_per_select={:.1}  total_ops_mean={:.0}",
        mean(t_eval.iter().map(|m| m.enemies_neutralized as f64), n),
        mean(t_eval.iter().map(|m| m.mean_survival_s), n),
        mean(t_eval.iter().map(|m| m.mean_compute_ops), n),
        mean(
            t_eval
                .iter()
                .map(|m| m.mean_compute_ops * m.decision_count as f64),
            n
        )
    );
    println!(
        "closer_than_friend  killed_mean={:.2}  survival_mean={:.4}  ops_per_select={:.1}  total_ops_mean={:.0}",
        mean(r_eval.iter().map(|m| m.enemies_neutralized as f64), n),
        mean(r_eval.iter().map(|m| m.mean_survival_s), n),
        mean(r_eval.iter().map(|m| m.mean_compute_ops), n),
        mean(
            r_eval
                .iter()
                .map(|m| m.mean_compute_ops * m.decision_count as f64),
            n
        )
    );
    println!(
        "plots: {}  {}",
        out_dir.join("plots/compute_ops_compare.svg").display(),
        out_dir.join("plots/survival_dist_compare.svg").display()
    );
    Ok(())
}

fn config_from_serde(e: serde_json::Error) -> RunError {
    RunError::Config(battlefield_sim::ConfigError::InvalidFormation {
        reason: format!("config json: {e}"),
    })
}
