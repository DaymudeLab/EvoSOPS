mod GACore;
mod SOPSCore;
use GACore::base_ga::GeneticAlgo;
use crate::SOPSCore::SOPSEnvironment;
use gag::Redirect;
use rand::SeedableRng;
use rand::{distributions::Bernoulli, distributions::Open01, distributions::Uniform, rngs, Rng};
use nanorand::{Rng as NanoRng, WyRand};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::time::Instant;
// use std::env;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::usize;
// use std::path::Path;
// use std::fs::File;
// use std::io::prelude::*;
// use chrono::prelude::*;

// const EXP: &'static str = "1";
// const ROOT_PATH: &'static str = "/output/";

fn get_temp_filepath() -> String {
    #[cfg(unix)]
    return "./output/trial.log".into();
}
// static mut genome_cache: Option<HashMap<[u16; 6], f64>> = None;

fn main() {
    // save a cache
    // unsafe {
    //     genome_cache = Some(HashMap::new());
    // }
    // Open a log
    let log = OpenOptions::new()
        .truncate(true)
        .read(true)
        .create(true)
        .write(true)
        .open(get_temp_filepath())
        .unwrap();

    let print_redirect = Redirect::stdout(log).unwrap();
    //size = [(6,4),(10,7),(13,9),(20,14),(27,19)]
    //seeds = [31728, 26812, 73921, 92031, 84621]
    // let mut ga_sops = GeneticAlgo::init_ga(2, 5, 0, 0.16, 20, true, [(6,4)].to_vec(), [31728].to_vec());
    // ga_sops.run_through();

    // /*
    // Block for running single experiments
    // Genome from know solution space -> 307
    // let genome = [
    //     0.879410923312898,
    //     0.7072823455008604,
    //     0.0758316160483933,
    //     0.0447528743281018,
    //     0.005321085020900764,
    //     0.0018268442926183,
    // ];
    // let genome = [
    //     0.5994110836839489,
    //     0.6183247501358494,
    //     0.5522009562182426,
    //     0.4959382596880117,
    //     0.2769269870929103,
    //     0.523770334862512,
    // ];
    let genome = [
        19, 20, 18, 1, 1, 1
    ];
    // println!("{:?}", genome);

    let mut sops_trial = SOPSEnvironment::init_sops_env(&genome, 6, 4, 31728);
    sops_trial.print_grid();
    let edge_cnt: u64 = sops_trial.evaluate_fitness().into();
    println!("Edge Count: {}", edge_cnt);
    println!("Max Edge Count: {}", sops_trial.get_max_edge_cnt());
    println!("Starting Fitness: {}", edge_cnt as f32/ sops_trial.get_max_edge_cnt() as f32);
    println!("No. of Participants {}", sops_trial.get_participant_cnt());
    let now = Instant::now();
    let edge_cnt: u64 = sops_trial.simulate(true).into();
    let elapsed = now.elapsed().as_secs();
    sops_trial.print_grid();
    println!("Edge Count: {}", edge_cnt);
    println!("Fitness: {}", edge_cnt as f32/ sops_trial.get_max_edge_cnt() as f32);
    println!("Trial Elapsed Time: {:.2?}", elapsed);
    // let scores: Vec<u32> = (0..10)
    //     .into_iter()
    //     .map(|idx| {
    //         let mut sops_trial_2 = SOPSEnvironment::static_init(&genome);
    //         // sops_trial_2.print_grid();
    //         // println!("{}", sops_trial_2.evaluate_fitness());
    //         // // println!("{}",sops_trial.evaluate_fitness());
    //         // println!("{}", );
    //         // sops_trial_2.print_grid();
    //         let now = Instant::now();
    //         let score = sops_trial_2.simulate();
    //         let elapsed = now.elapsed().as_secs();
    //         println!("Trial {idx} Elapsed Time: {:.2?}", elapsed);
    //         score
    //     })
    //     .collect();

    // let mean = scores.clone().into_iter().fold(0, |sum, score| sum + score) / 1;

    // let variance = scores
    //     .into_iter()
    //     .fold(0, |sum, score| sum + (score as i32 - mean as i32).pow(2))
    //     / 1;

    // println!("N^3 Mean:{}", mean);
    // println!("N^3 Variance:{}", variance);
    // scores.for_each(|s| { println!("{}", s);});
    //  */
    // let mut grid: [[bool; 18]; 18] = [[false; 18]; 18];
    // let init_nrm = Normal::new(0.0, 0.05).unwrap();
    // let bi = Bernoulli::new(0.5).unwrap();
    // let mut rng = rand::thread_rng();
    // let grid_rng = Uniform::new(0, grid.len());
    // let mut cnt = 0;
    // for _i in 0..1000 {
    //     let v = rng.sample(&bi);
    //     // println!("{}", v);
    //     if v { cnt += 1; }
    // }
    // println!("{}",cnt);
    // println!("{} particles in grid of {} vertices",(((grid.len()*grid.len()) as f32)*0.3) as u64, grid.len()*grid.len());
    print_redirect.into_inner();
}
