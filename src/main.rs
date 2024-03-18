mod GACore;
mod SOPSCore;
mod utils;

use GACore::brid_ga::BridGA;

use crate::SOPSCore::bridging::SOPSBridEnvironment;

use clap::{Parser, ValueEnum};
use gag::Redirect;
use rayon::prelude::*;
use std::fs;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::time::Instant;
use std::usize;

fn get_temp_filepath(filename: &String) -> String {
    #[cfg(unix)]
    return "./output/".to_string() + filename + &".log".to_string();
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Behaviour to run
    #[arg(short, long, value_enum)]
    behavior: Behavior,

    /// Type of Experiment to run
    #[arg(short, long = "exp", value_enum)]
    experiment_type: Experiment,

    /// Maximum no. of generations to run the Genetic Algorithm Experiment
    #[arg(short = 'g', long = "gen", default_value_t = 0)]
    max_generations: u16,

    /// No. of genomes in the population for a Genetic Algorithm Experiment
    #[arg(short = 'p', long = "pop", default_value_t = 0)]
    population: u16,

    /// Genome representation granularity
    #[arg(long = "gran", default_value_t = 20)]
    granularity: u8,

    /// Mutation rate per gene
    #[arg(short = 'm', long = "mut", default_value_t = 0.08)]
    mutation_rate: f64,

    /// Maximum no. of elite genomes to preserve in a generation
    #[arg(long = "eli", default_value_t = 0)]
    elitist_count: u16,

    /// Particle Sizes to run on (eg. use multiple -k<String"(<u64>,<u64>)"> arguments to specify multiple sizes)
    #[arg(short='k', long="ks", required=true, action = clap::ArgAction::Append)]
    particle_sizes: Vec<String>,

    /// Seed values to run experiments with, for reproducable trials (eg. use multiple -s<u64> arguments to specify seeded trials)
    #[arg(short, long="seed", required=true, action = clap::ArgAction::Append)]
    seeds: Vec<u64>,

    /// File to read genome value from
    #[arg(long, value_name = "FILE")]
    path: Option<PathBuf>,

    /// Specify if execution description is written to the output
    #[arg(short, long)]
    verbose: bool,

    /// Specify if snapshots of the experiment run are written to the output (only valid for stand-alone experiments)
    #[arg(long)]
    snaps: bool,
}

#[derive(ValueEnum, Debug, Clone)] // ArgEnum here
enum Behavior {
    /// Aggregation
    Agg,
    /// Separation
    Sep,
    /// Bridging
    Brid,
}

#[derive(ValueEnum, Debug, Clone)] // ArgEnum here
enum Experiment {
    /// Full scale genetic algorithm run
    GA,
    /// Stand-alone single genome run
    GM,
    /// Stand-alone single theory given solution run
    TH,
}

fn main() {
    let args = Args::parse();
    /*
     * Pipe the output to the file with Experiment Parameters as its name
     */
    let file_name = format!(
        "{:?}_{:?}_{}_sizes_{}_trials_gran_{}",
        &args.behavior,
        &args.experiment_type,
        &args.particle_sizes.len(),
        &args.seeds.len(),
        &args.granularity
    );
    println!(
        "Running the experiment... \nPlease check: {:?} file in ./output folder",
        &file_name
    );

    let log = OpenOptions::new()
        .truncate(true)
        .read(true)
        .create(true)
        .write(true)
        .open(get_temp_filepath(&file_name))
        .unwrap();

    let print_redirect = Redirect::stdout(log).unwrap();

    /*
     * Print out the options for the current run of the script
     */
    println!("Target Behavior: {:?}", &args.behavior);
    println!("Experiment Type: {:?}", &args.experiment_type);
    println!("Particle Sizes: {:?}", &args.particle_sizes);
    println!("Initialization Seeds: {:?}", &args.seeds);
    println!("Representation Granularity: {:?}", &args.granularity);

    /*
     * Convert Particle Sizes from Vec<String> to Vec<(u16,u16)>
     */
    let size_strings = args.particle_sizes.iter().map(|s| {
        s.split(&['(', ')', ','][..])
            .filter_map(|ss| ss.parse::<u16>().ok())
            .collect::<Vec<u16>>()
    });
    let particle_sizes: Vec<(u16, u16)> = size_strings
        .map(|c| (c[0], c[1]))
        .collect::<Vec<(u16, u16)>>();

    /*
     * Based on Experiment type and Behaviour setup required experiment parameters
     */
    match &args.experiment_type {
        Experiment::GA => {
            let crossover = true;
            println!("Population Size: {:?}", &args.population);
            println!("Max Generations: {:?}", &args.max_generations);
            println!("Mutation Rate: {:?}", &args.mutation_rate);
            println!("Elitist Count: {:?}", &args.elitist_count);
            println!("Cross-over: {:?}", &crossover);
            /*
             * Perform a single run of Full length Genetic algorithm for respective behaviour
             */
            match &args.behavior {
                Behavior::Agg => {
                    todo!();
                }
                Behavior::Sep => {
                    todo!();
                }
                Behavior::Brid => {
                    println!("\nStarting Bridging GA Experiment...\n");
                    let mut ga_sops = BridGA::init_ga(
                        args.population,
                        args.max_generations,
                        args.elitist_count,
                        args.mutation_rate,
                        args.granularity,
                        true,
                        particle_sizes,
                        args.seeds,
                    );
                    ga_sops.run_through();
                }
            }
        }
        ref other_experiment => {
            println!("Snapshots: {:?}", &args.snaps);

            assert_eq!(&args.path.is_some(), &true);

            let path = &args.path.clone().unwrap();

            assert_eq!(&path.is_file(), &true);

            println!("Genome file path: {}", &path.display());
            // Read Genome file content and pre-process it
            let contents =
                fs::read_to_string(&path).expect("Should have been able to read the file");
            let mut striped_content = contents.replace("[", "").replace("]", "").replace(" ", "");
            striped_content.pop();

            /*
             * Perform a standalone evaluation runs of given Genomes for respective behaviours
             * Total Runs = #. of Particle Sizes x #. of Seeds
             */
            match &other_experiment {
                Experiment::GM => {
                    let all_entries: Vec<u8> = striped_content
                        .split(',')
                        .filter_map(|x| x.parse::<u8>().ok())
                        .collect();

                    match &args.behavior {
                        Behavior::Agg => {
                            println!("\nStarting Aggregation Single Genome Trial...\n");
                            // Construct the genome in required dimension
                            todo!();
                        }
                        Behavior::Sep => {
                            println!("\nStarting Separation Single Genome Trial...\n");
                            // Construct the genome in required dimension
                            todo!();
                        }
                        Behavior::Brid => {
                            println!("\nStarting Bridging Single Genome Trial...\n");

                            let genome: [[[[u8; 2]; 10]; 6]; 10] = [[[[10, 1], [9, 0], [1, 8], [0, 6], [1, 1], [0, 8], [1, 2], [6, 0], [7, 1], [0, 2]], [[7, 8], [7, 8], [3, 0], [0, 6], [3, 0], [7, 1], [1, 6], [5, 3], [2, 0], [8, 3]], [[7, 5], [2, 1], [9, 0], [5, 1], [0, 0], [0, 3], [0, 7], [1, 0], [10, 0], [6, 1]], [[4, 0], [0, 10], [1, 3], [3, 3], [5, 0], [5, 7], [6, 3], [0, 5], [1, 6], [7, 0]], [[6, 1], [10, 0], [0, 0], [1, 6], [3, 4], [1, 0], [4, 0], [3, 0], [9, 1], [0, 9]], [[0, 2], [1, 5], [0, 3], [6, 0], [5, 3], [3, 5], [4, 1], [5, 0], [5, 4], [0, 5]]], [[[4, 7], [2, 5], [1, 8], [5, 7], [0, 5], [0, 0], [6, 0], [7, 1], [1, 10], [0, 2]], [[6, 6], [7, 6], [0, 0], [4, 1], [0, 5], [8, 0], [6, 5], [2, 1], [1, 5], [4, 9]], [[5, 5], [1, 6], [6, 9], [1, 6], [0, 3], [2, 3], [3, 0], [0, 0], [8, 10], [1, 8]], [[7, 3], [4, 7], [3, 7], [5, 1], [7, 0], [6, 6], [7, 5], [4, 0], [0, 5], [6, 4]], [[3, 5], [0, 4], [2, 0], [0, 0], [3, 10], [0, 3], [5, 10], [3, 5], [5, 5], [0, 1]], [[4, 5], [0, 0], [4, 2], [0, 6], [4, 5], [4, 5], [0, 4], [0, 5], [8, 6], [0, 4]]], [[[9, 0], [2, 3], [7, 1], [4, 9], [5, 3], [0, 0], [3, 0], [6, 5], [0, 7], [1, 3]], [[4, 7], [5, 6], [8, 7], [2, 5], [6, 3], [6, 4], [0, 4], [5, 0], [4, 5], [8, 4]], [[9, 5], [4, 1], [9, 9], [2, 4], [1, 3], [8, 6], [4, 0], [3, 0], [1, 8], [8, 0]], [[6, 9], [3, 0], [5, 8], [7, 1], [8, 1], [3, 6], [6, 1], [4, 1], [3, 0], [6, 5]], [[6, 8], [5, 3], [8, 7], [8, 3], [3, 1], [7, 0], [8, 7], [4, 2], [6, 0], [2, 0]], [[6, 1], [2, 3], [1, 7], [0, 0], [0, 1], [9, 6], [7, 4], [0, 1], [2, 8], [7, 0]]], [[[5, 4], [2, 4], [1, 8], [0, 3], [4, 10], [2, 0], [9, 9], [3, 0], [2, 2], [1, 6]], [[9, 3], [3, 0], [0, 8], [8, 0], [1, 8], [8, 3], [5, 0], [5, 7], [1, 2], [4, 9]], [[0, 6], [0, 4], [3, 0], [4, 0], [6, 6], [0, 5], [4, 1], [6, 2], [3, 3], [0, 7]], [[8, 4], [5, 1], [6, 4], [4, 3], [0, 1], [0, 0], [2, 5], [8, 3], [4, 2], [10, 7]], [[1, 7], [0, 2], [2, 3], [0, 1], [10, 3], [0, 4], [1, 4], [2, 2], [5, 0], [0, 7]], [[1, 5], [3, 3], [1, 0], [6, 1], [6, 8], [2, 0], [8, 1], [0, 9], [1, 2], [1, 0]]], [[[8, 10], [3, 7], [4, 3], [7, 1], [3, 5], [4, 1], [10, 6], [9, 9], [9, 6], [0, 1]], [[4, 8], [7, 0], [1, 2], [8, 2], [1, 9], [1, 7], [0, 6], [8, 8], [3, 7], [4, 6]], [[5, 8], [4, 2], [10, 0], [6, 6], [4, 3], [1, 4], [4, 0], [8, 8], [0, 0], [0, 0]], [[5, 8], [6, 4], [4, 5], [2, 1], [9, 8], [2, 1], [8, 0], [8, 8], [0, 7], [6, 3]], [[7, 3], [0, 7], [4, 0], [4, 10], [0, 4], [3, 1], [5, 8], [3, 7], [1, 2], [5, 1]], [[0, 10], [2, 7], [0, 7], [0, 3], [3, 0], [7, 5], [1, 8], [6, 7], [1, 3], [4, 5]]], [[[10, 5], [1, 2], [9, 1], [7, 1], [4, 5], [0, 4], [9, 6], [3, 5], [0, 1], [6, 9]], [[5, 4], [5, 1], [3, 8], [5, 6], [5, 10], [5, 4], [4, 8], [5, 4], [7, 6], [2, 4]], [[10, 1], [6, 2], [10, 8], [4, 2], [10, 1], [8, 0], [8, 3], [1, 3], [0, 1], [6, 1]], [[5, 0], [4, 2], [0, 8], [6, 4], [3, 7], [6, 8], [8, 5], [1, 2], [10, 1], [9, 7]], [[8, 9], [10, 1], [6, 4], [5, 3], [7, 2], [8, 2], [1, 1], [7, 4], [1, 0], [10, 8]], [[10, 2], [8, 7], [0, 5], [4, 8], [7, 0], [2, 0], [5, 0], [0, 3], [1, 2], [6, 3]]], [[[4, 10], [4, 1], [8, 3], [2, 4], [4, 1], [10, 3], [7, 8], [1, 0], [2, 10], [6, 0]], [[5, 6], [7, 4], [0, 8], [3, 5], [9, 7], [6, 9], [0, 5], [0, 5], [0, 5], [0, 5]], [[0, 0], [1, 1], [3, 5], [6, 0], [0, 0], [1, 0], [0, 3], [5, 8], [2, 4], [0, 7]], [[3, 8], [7, 1], [10, 2], [8, 3], [1, 1], [5, 1], [3, 6], [1, 1], [9, 6], [2, 6]], [[1, 4], [7, 3], [7, 9], [4, 3], [2, 0], [7, 3], [7, 6], [0, 3], [2, 0], [4, 2]], [[1, 6], [0, 6], [2, 5], [0, 4], [3, 5], [4, 6], [0, 1], [3, 5], [0, 4], [0, 3]]], [[[5, 7], [3, 0], [2, 7], [2, 2], [1, 0], [0, 0], [2, 1], [3, 4], [7, 6], [0, 5]], [[8, 2], [7, 6], [8, 0], [9, 1], [0, 3], [8, 0], [0, 0], [3, 1], [1, 9], [2, 1]], [[5, 5], [9, 4], [7, 1], [1, 4], [6, 1], [5, 0], [0, 5], [2, 0], [1, 5], [1, 2]], [[0, 2], [0, 2], [0, 0], [6, 10], [0, 3], [5, 5], [0, 2], [5, 6], [0, 1], [9, 2]], [[7, 0], [8, 5], [9, 0], [6, 2], [7, 1], [6, 2], [0, 9], [9, 5], [4, 1], [5, 5]], [[2, 9], [7, 5], [2, 5], [3, 0], [1, 0], [3, 7], [0, 0], [7, 5], [4, 0], [3, 7]]], [[[5, 4], [3, 3], [0, 0], [3, 1], [7, 8], [4, 0], [1, 9], [1, 9], [8, 0], [2, 7]], [[8, 8], [6, 0], [2, 1], [4, 0], [1, 5], [6, 5], [0, 0], [0, 2], [2, 7], [4, 2]], [[1, 3], [3, 1], [10, 5], [1, 4], [10, 3], [0, 1], [1, 0], [6, 7], [2, 0], [5, 3]], [[4, 6], [8, 0], [6, 9], [5, 6], [4, 1], [10, 10], [0, 2], [8, 5], [2, 8], [6, 2]], [[1, 8], [4, 7], [4, 0], [7, 0], [2, 0], [0, 3], [4, 0], [6, 5], [2, 4], [1, 4]], [[5, 9], [3, 3], [8, 0], [8, 3], [5, 8], [5, 7], [4, 1], [2, 4], [1, 6], [2, 5]]], [[[10, 0], [7, 2], [3, 0], [3, 0], [10, 3], [0, 2], [1, 1], [1, 2], [1, 7], [3, 7]], [[8, 0], [9, 2], [7, 1], [7, 8], [2, 5], [2, 2], [10, 6], [8, 10], [6, 7], [2, 4]], [[9, 0], [9, 1], [3, 0], [7, 2], [4, 5], [0, 4], [0, 0], [1, 1], [0, 8], [0, 0]], [[8, 2], [8, 2], [5, 5], [2, 9], [6, 4], [4, 7], [10, 0], [8, 2], [0, 0], [5, 0]], [[6, 0], [6, 2], [1, 4], [4, 0], [8, 4], [4, 4], [1, 0], [5, 0], [1, 0], [2, 1]], [[9, 4], [6, 5], [0, 1], [0, 1], [0, 3], [0, 9], [4, 4], [0, 4], [5, 9], [0, 2]]]];

                            // let mut genome: [[[[u8; 2]; 10]; 6]; 10] = [[[[0; 2]; 10]; 6]; 10];
                            // let mut idx = 0;
                            // println!("All entries: ");
                            // let mut entry_amount = 0;
                            // for i in &all_entries {
                            //     println!(" {} ", i);
                            //     entry_amount += 1;
                            // }

                            // println!("Amount of entries: {}", entry_amount);

                            // for n in 0_u8..10 {
                            //     for j in 0_u8..6 {
                            //         for i in 0_u8..10 {
                            //             for k in 0_u8..2 {
                            //                 genome[n as usize][j as usize][i as usize][k as usize] =
                            //                     all_entries[idx];
                            //                 idx += 1;
                            //             }
                            //         }
                            //     }
                            // }

                            println!("Read Genome:\n{:?}", genome);

                            let trials = args.seeds.len();
                            let seeds = args.seeds.clone();

                            let trials_vec: Vec<((u16, u16), u64)> = particle_sizes
                                .clone()
                                .into_iter()
                                .zip(seeds)
                                .flat_map(|v| std::iter::repeat(v).take(trials.into()))
                                .collect();

                            let fitness_tot: f64 = trials_vec
                                .clone()
                                .into_par_iter()
                                .map(|trial| {
                                    /*
                                     * Single Evaluation run of the Genome
                                     */
                                    let mut sops_trial = SOPSBridEnvironment::init_sops_env(
                                        &genome,
                                        13,
                                        3,
                                        17,
                                        trial.1,
                                        args.granularity,
                                    );
                                    sops_trial.print_grid();
                                    let edge_cnt: f32 = sops_trial.evaluate_fitness();
                                    println!("Edge Count: {}", edge_cnt);
                                    println!("Max Fitness: {}", sops_trial.get_max_fitness());
                                    println!(
                                        "Starting Fitness: {}",
                                        edge_cnt as f32 / sops_trial.get_max_fitness() as f32
                                    );
                                    let now = Instant::now();
                                    let edge_cnt: f32 = sops_trial.simulate(true);
                                    let elapsed = now.elapsed().as_secs();
                                    sops_trial.print_grid();
                                    println!("Edge Count: {}", edge_cnt);
                                    let t_fitness =
                                        edge_cnt as f64 / sops_trial.get_max_fitness() as f64;
                                    println!("Fitness: {}", &t_fitness);
                                    println!("Trial Elapsed Time: {:.2?}s", elapsed);
                                    t_fitness
                                })
                                .sum();

                            println!("Total Fitness: {}", &fitness_tot);
                        }
                    }
                }
                Experiment::TH => {
                    let all_entries: Vec<f32> = striped_content
                        .split(',')
                        .map(|x| x.parse::<f32>().unwrap())
                        .collect();

                    match &args.behavior {
                        Behavior::Agg => {
                            let mut genome: [f32; 6] = [0.0; 6];
                            let mut idx = 0;
                            for i in 0_u8..6 {
                                genome[i as usize] = all_entries[idx];
                                idx += 1;
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        }
                        Behavior::Sep => {
                            let mut genome: [[[f32; 6]; 7]; 7] = [[[0.0; 6]; 7]; 7];
                            let mut idx = 0;
                            for n in 0_u8..7 {
                                for j in 0_u8..7 {
                                    for i in 0_u8..6 {
                                        // if i+j <= n {
                                        genome[n as usize][j as usize][i as usize] =
                                            all_entries[idx];
                                        idx += 1;
                                        // }
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        }
                        Behavior::Brid => {
                            todo!()
                        }
                    }
                }
                _ => {}
            }
        }
    }

    print_redirect.into_inner();
}
