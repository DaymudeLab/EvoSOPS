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

fn get_temp_filepath(filename: &String, output_path: &Option<String>) -> String {
    if let Some(output_path) = output_path {
        #[cfg(unix)]
        return output_path.to_string() + filename + &".log".to_string();
    } else {
        #[cfg(unix)]
        return "./output/".to_string() + filename + &".log".to_string();
    }
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

    /// Particle Sizes to run on (eg. use multiple -k<String"(<u64>,<u64>,<u64>,<u64>)"> arguments to specify multiple sizes)
    #[arg(short='k', long="ks", required=true, action = clap::ArgAction::Append)]
    arena_parameters: Vec<String>,

    /// Weights for experiment to run (eg. -w<String"(<f32>,<f32>,<f32>,<f32>)">)
    #[arg(short='w', long="wf", required=true, action = clap::ArgAction::Append)]
    weights: String,

    /// Length of theory run
    #[arg(short='d', required=false, long="dur", default_value_t = 8000000)]
    duration: u32,

    /// File to write output to
    // #[arg(long, required=false, value_name = "FILE")]
    // path: Option<PathBuf>,

    /// File to write output to
    #[arg(long, required=false)]
    output_path: Option<String>,

    /// Specify if execution description is written to the output
    #[arg(short, long)]
    verbose: bool,

    /// Specify if snapshots of the experiment run are written to the output (only valid for stand-alone experiments)
    #[arg(long)]
    snaps: bool,

    /// Specify trial number
    #[arg(short='t', required=false, long="trial", default_value_t = 0)]
    trial_number: u32,
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
        "{:?}_{:?}_{}_sizes_trials_gran_{}_trial_{}",
        &args.behavior,
        &args.experiment_type,
        &args.arena_parameters.len(),
        &args.granularity,
        &args.trial_number
    );
    println!(
        "Running the experiment... \nPlease check: {:?} file in {:?} folder",
        &file_name,
        if let Some(output_path) = &args.output_path {
            output_path.to_string()
        } else {
            "./output".to_string()
        }
    );

    let log = OpenOptions::new()
        .truncate(true)
        .read(true)
        .create(true)
        .write(true)
        .open(get_temp_filepath(&file_name, &args.output_path))
        .unwrap();

    let print_redirect = Redirect::stdout(log).unwrap();

    /*
     * Print out the options for the current run of the script
     */
    println!("Target Behavior: {:?}", &args.behavior);
    println!("Experiment Type: {:?}", &args.experiment_type);
    println!("Arena Parameters: {:?}", &args.arena_parameters);
    println!("Weights: {:?}", &args.weights);
    println!("Representation Granularity: {:?}", &args.granularity);

    /*
     * Convert Particle Sizes from Vec<String> to Vec<(u16,u16)>
     */
    
    let size_strings = args.arena_parameters.iter().map(|s| {
        s.split(&['(', ')', ','])
            .filter_map(|ss| ss.parse::<u16>().ok())
            .collect::<Vec<u16>>()
    });
     
    let arena_parameters: Vec<(u16, u16, u16, u16)> = size_strings
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect::<Vec<(u16, u16, u16, u16)>>();

    let weights_vec = args.weights.split(&['(', ')', ','][..])
    .filter_map(|ss| ss.parse::<f32>().ok())
    .collect::<Vec<f32>>();
    let weights: (f32, f32, f32, f32);

    if weights_vec.len() != 4 {
        panic!("Too many weights.");
    } else {
        weights = (weights_vec[0], weights_vec[1], weights_vec[2], weights_vec[3]);
    }

    /*
     * Based on Experiment type and Behaviour setup required experiment parameters
     */
    match &args.experiment_type {
        Experiment::GA => {
            println!("Population Size: {:?}", &args.population);
            println!("Max Generations: {:?}", &args.max_generations);
            println!("Mutation Rate: {:?}", &args.mutation_rate);
            println!("Elitist Count: {:?}", &args.elitist_count);
            println!("Cross-over: {:?}", true);
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
                        arena_parameters,
                        weights
                    );
                    ga_sops.run_through();
                }
            }
        }
        ref other_experiment => {
            println!("Snapshots: {:?}", &args.snaps);

            // assert_eq!(&args.path.is_some(), &true);

            // let path = &args.path.clone().unwrap();

            // assert_eq!(&path.is_file(), &true);

            // println!("Genome file path: {}", &path.display());
            // // Read Genome file content and pre-process it
            // let contents =
            //     fs::read_to_string(&path).expect("Should have been able to read the file");
            // let mut striped_content = contents.replace("[", "").replace("]", "").replace(" ", "");
            // striped_content.pop();

            /*
             * Perform a standalone evaluation runs of given Genomes for respective behaviours
             * Total Runs = #. of Particle Sizes x #. of Seeds
             */
            match &other_experiment {
                Experiment::GM => {
                    // let all_entries: Vec<u8> = striped_content
                    //     .split(',')
                    //     .filter_map(|x| x.parse::<u8>().ok())
                    //     .collect();

                    match &args.behavior {
                        Behavior::Agg => {
                            println!("\nStarting Aggregation Single Genome Trial...\n");
                            todo!();
                        }
                        Behavior::Sep => {
                            println!("\nStarting Separation Single Genome Trial...\n");
                            todo!();
                        }
                        Behavior::Brid => {
                            println!("\nStarting Bridging Single Genome Trial...\n");

                            let genome: [[[[[u8; 3]; 2]; 4]; 3]; 4] =  [[[[[8, 10, 4], [7, 4, 8]], [[0, 8, 9], [3, 8, 9]], [[0, 6, 5], [5, 2, 3]], [[6, 5, 1], [4, 6, 7]]], [[[8, 10, 0], [9, 9, 4]], [[5, 4, 3], [2, 9, 2]], [[7, 6, 2], [2, 3, 2]], [[1, 8, 0], [8, 4, 2]]], [[[5, 7, 0], [5, 9, 4]], [[8, 8, 4], [2, 2, 6]], [[2, 1, 1], [5, 7, 5]], [[5, 3, 3], [8, 6, 6]]]], [[[[10, 9, 5], [10, 4, 8]], [[9, 2, 4], [7, 3, 10]], [[1, 5, 6], [5, 9, 0]], [[1, 9, 1], [7, 5, 6]]], [[[10, 10, 0], [7, 4, 6]], [[7, 10, 2], [4, 0, 3]], [[10, 10, 0], [4, 9, 8]], [[8, 6, 1], [2, 9, 8]]], [[[10, 6, 10], [4, 2, 5]], [[5, 9, 1], [0, 8, 4]], [[4, 7, 1], [3, 6, 0]], [[8, 1, 1], [7, 8, 9]]]], [[[[10, 10, 9], [8, 8, 9]], [[9, 5, 4], [8, 7, 9]], [[2, 2, 8], [1, 8, 5]], [[5, 6, 4], [10, 5, 2]]], [[[10, 10, 2], [10, 9, 6]], [[7, 9, 3], [2, 7, 9]], [[6, 6, 2], [5, 1, 10]], [[5, 5, 8], [9, 4, 3]]], [[[10, 10, 10], [10, 9, 7]], [[1, 3, 8], [9, 9, 4]], [[10, 8, 1], [1, 10, 8]], [[10, 4, 3], [3, 5, 9]]]], [[[[10, 8, 5], [8, 5, 6]], [[4, 8, 6], [1, 2, 3]], [[6, 4, 3], [6, 1, 7]], [[10, 8, 6], [5, 9, 10]]], [[[9, 8, 5], [8, 9, 3]], [[10, 6, 8], [3, 5, 9]], [[2, 6, 2], [8, 8, 3]], [[10, 8, 1], [9, 4, 2]]], [[[6, 9, 6], [9, 10, 8]], [[2, 9, 0], [1, 6, 9]], [[8, 8, 1], [6, 2, 5]], [[7, 10, 5], [1, 3, 8]]]]];

                            // let genome: [[[[[u8; 3]; 2]; 4]; 3]; 4] = [[[[[0; 3]; 2]; 4]; 3]; 4];
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

                            let trials_vec: Vec<(u16, u16, u16, u16)> = arena_parameters.clone().into_iter()
                            .flat_map(|i| std::iter::repeat(i).take(1))
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
                                        trial.0,
                                        trial.1,
                                        trial.2,
                                        trial.3,
                                        args.granularity,
                                        weights.0,
                                        weights.1,
                                        weights.2,
                                        weights.3
                                    );
                                    sops_trial.print_grid();
                                    let edge_cnt: f32 = sops_trial.evaluate_fitness();
                                    println!("Edge Count: {}", edge_cnt);
                                    println!("Max Fitness: {}", sops_trial.get_max_fitness());
                                    println!(
                                        "Starting Fitness: {}",
                                        (edge_cnt as f32)
                                    );
                                    let now = Instant::now();
                                    let edge_cnt: f32 = sops_trial.simulate(true);
                                    let elapsed = now.elapsed().as_secs();
                                    sops_trial.print_grid();
                                    println!("Edge Count: {}", edge_cnt);
                                    let t_fitness = edge_cnt as f64;
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
                    match &args.behavior {
                        Behavior::Agg => {
                            // let mut genome: [f32; 6] = [0.0; 6];
                            // let mut idx = 0;
                            // for i in 0_u8..6 {
                            //     genome[i as usize] = all_entries[idx];
                            //     idx += 1;
                            // }

                            // println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        }
                        Behavior::Sep => {
                            // let mut genome: [[[f32; 6]; 7]; 7] = [[[0.0; 6]; 7]; 7];
                            // let mut idx = 0;
                            // for n in 0_u8..7 {
                            //     for j in 0_u8..7 {
                            //         for i in 0_u8..6 {
                            //             // if i+j <= n {
                            //             genome[n as usize][j as usize][i as usize] =
                            //                 all_entries[idx];
                            //             idx += 1;
                            //             // }
                            //         }
                            //     }
                            // }

                            // println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        }
                        Behavior::Brid => {
                            let genome: [[[[[u8; 3]; 2]; 4]; 3]; 4] = [[[[[0; 3]; 2]; 4]; 3]; 4];

                            let trials_vec: Vec<(u16, u16, u16, u16)> = arena_parameters.clone().into_iter()
                            .flat_map(|i| std::iter::repeat(i).take(3))
                            .collect();

                            let fitness_tot: f64 = trials_vec
                                .clone()
                                .into_iter()
                                .map(|trial| {
                                    /*
                                     * Single Evaluation run of the Genome
                                     */
                                    println!("Trial Info: {}, {}, {}, {}", trial.0, trial.1, trial.2, trial.3);
                                    let mut sops_trial = SOPSBridEnvironment::init_sops_env(
                                        &genome,
                                        trial.0,
                                        trial.1,
                                        trial.2,
                                        trial.3,
                                        args.granularity,
                                        weights.0,
                                        weights.1,
                                        weights.2,
                                        weights.3
                                    );
                                    sops_trial.print_grid();
                                    let edge_cnt: f32 = sops_trial.evaluate_fitness();
                                    println!("Edge Count: {}", edge_cnt);
                                    println!("Max Fitness: {}", sops_trial.get_max_fitness());
                                    println!(
                                        "Starting Fitness: {}",
                                        edge_cnt as f32 as f32
                                    );
                                    let now = Instant::now();
                                    let edge_cnt: f32 = sops_trial.simulate_theory(args.duration as u64,false);
                                    let elapsed = now.elapsed().as_secs();
                                    sops_trial.print_grid();
                                    println!("Edge Count: {}", edge_cnt);
                                    let t_fitness =
                                        edge_cnt as f64 as f64;
                                    println!("Fitness: {}", &t_fitness);
                                    println!("Trial Elapsed Time: {:.2?}s", elapsed);
                                    t_fitness
                                })
                                .sum();

                            println!("Total Fitness: {}", &fitness_tot / trials_vec.len() as f64);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    print_redirect.into_inner();
}
