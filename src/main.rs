mod GACore;
mod SOPSCore;
use GACore::base_ga::GeneticAlgo;
use GACore::seg_ga::SegGA;
use crate::SOPSCore::SOPSEnvironment;
use crate::SOPSCore::segregation::SOPSegEnvironment;
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

fn get_temp_filepath(trial_seed: u64) -> String {
    #[cfg(unix)]
    return "./output/trial_".to_string() + &trial_seed.to_string() + &".log".to_string();
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
        .open(get_temp_filepath(fastrand::Rng::new().u64(1_u64..=u64::MAX)))
        .unwrap();

    let print_redirect = Redirect::stdout(log).unwrap();
    //size = [(6,4),(10,7),(13,9),(20,14),(27,19)]
    //seeds = [31728, 26812, 73921, 92031, 84621]
    // let mut ga_sops = SegGA::init_ga(50, 100, 1, 0.08, 10, true, [(10,7)].to_vec(), [31728, 26812, 73921].to_vec());
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
    // let genome = [[17, 0, 0, 0, 0, 0, 0], [12, 20, 0, 0, 0, 0, 0], [10, 3, 15, 0, 0, 0, 0], [7, 8, 13, 2, 0, 0, 0], [1, 20, 16, 14, 2, 0, 0], [1, 16, 9, 2, 7, 13, 0], [4, 20, 3, 12, 11, 15, 7]];
    // let genome = [[16, 0, 0, 0, 0, 0, 0], [20, 20, 0, 0, 0, 0, 0], [15, 14, 13, 0, 0, 0, 0], [18, 1, 3, 9, 0, 0, 0], [12, 1, 5, 1, 2, 0, 0], [17, 4, 20, 1, 2, 7, 0], [4, 10, 4, 1, 14, 4, 14]];
    // let genome = [[16, 0, 0, 0, 0, 0], [20, 20, 0, 0, 0, 0], [15, 14, 13, 0, 0, 0], [18, 1, 3, 9, 0, 0], [12, 1, 5, 1, 2, 0], [17, 4, 20, 1, 2, 7], [4, 10, 4, 1, 14, 4]];
    // given by theory
    // let genome = [[20, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0], [2, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
    // MY TWEAKS
    // let genome = [[20, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0], [2, 2, 1, 0, 0, 0], [2, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]];
    // let genome = [[20, 0, 0, 0, 0, 0], [13, 1, 0, 0, 0, 0], [12, 5, 1, 0, 0, 0], [10, 5, 1, 0, 0, 0], [10, 5, 0, 0, 0, 0], [10, 6, 0, 0, 0, 0], [10, 6, 0, 0, 0, 0]];
    
    let genome = [[20, 0, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0], [2, 2, 1, 0, 0, 0], [2, 1, 1, 0, 0, 0], [2, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]];
    
    println!("{:?}", genome);

    let mut sops_trial = SOPSegEnvironment::init_sops_env(&genome, 10, 7, 31728);
    sops_trial.print_grid();
    let edge_cnt: u64 = sops_trial.evaluate_fitness().into();
    println!("Edge Count: {}", edge_cnt);
    println!("Max Fitness: {}", sops_trial.get_max_fitness());
    println!("Starting Fitness: {}", edge_cnt as f32/ sops_trial.get_max_fitness() as f32);
    println!("No. of Participants {:?}", sops_trial.get_participant_cnt());
    let now = Instant::now();
    let edge_cnt: u64 = sops_trial.simulate(true).into();
    let elapsed = now.elapsed().as_secs();
    sops_trial.print_grid();
    println!("Edge Count: {}", edge_cnt);
    println!("Fitness: {}", edge_cnt as f32/ sops_trial.get_max_fitness() as f32);
    println!("No. of Participants {:?}", sops_trial.get_participant_cnt());
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
