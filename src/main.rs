mod GACore;
mod SOPSCore;
mod utils;
use GACore::base_ga::GeneticAlgo;
use GACore::seg_ga::SegGA;
use GACore::coat_ga::CoatGA;

use crate::SOPSCore::SOPSEnvironment;
use crate::SOPSCore::segregation::SOPSegEnvironment;
use crate::SOPSCore::coating::SOPSCoatEnvironment;

use rayon::prelude::*;
use gag::Redirect;
use std::path::PathBuf;
use std::time::Instant;
use std::fs;
use std::fs::OpenOptions;
use std::usize;
use clap::{Parser, ValueEnum};

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
    #[arg(short, long="exp", value_enum)]
    experiment_type: Experiment,

    /// Maximum no. of generations to run the Genetic Algorithm Experiment
    #[arg(short='g', long="gen", default_value_t=0)]
    max_generations: u16,

    /// No. of genomes in the population for a Genetic Algorithm Experiment
    #[arg(short='p', long="pop", default_value_t=0)]
    population: u16,

    /// Genome representation granularity
    #[arg(long="gran", default_value_t=20)]
    granularity: u8,

    /// Mutation rate per gene
    #[arg(short='m', long="mut", default_value_t=0.08)]
    mutation_rate: f64,

    /// Maximum no. of elite genomes to preserve in a generation
    #[arg(long="eli", default_value_t=0)]
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
    /// Coating
    Coat,
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
    let file_name = format!("{:?}_{:?}_{}_sizes_{}_trials_gran_{}", &args.behavior, &args.experiment_type, &args.particle_sizes.len(), &args.seeds.len(), &args.granularity);
    println!("Running the experiment... \nPlease check: {:?} file in ./output folder", &file_name);
    
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
    let size_strings = args.particle_sizes
        .iter()
        .map(|s| s.split(&['(', ')', ','][..])
        .filter_map(|ss| 
            ss.parse::<u16>().ok()
        ).collect::<Vec<u16>>());
    let particle_sizes: Vec<(u16,u16)> = size_strings.clone().map(|c| (c[0],c[1])).collect::<Vec<(u16,u16)>>();

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
                    println!("\nStarting Aggregation GA Experiment...\n");
                    let mut ga_sops = GeneticAlgo::init_ga(args.population, args.max_generations,args.elitist_count, args.mutation_rate, args.granularity, crossover, particle_sizes, args.seeds);
                    ga_sops.run_through();
                },
                Behavior::Sep => {
                    println!("\nStarting Separation GA Experiment...\n");
                    let mut ga_sops = SegGA::init_ga(args.population, args.max_generations, args.elitist_count, args.mutation_rate, args.granularity, true, particle_sizes, args.seeds, 0.65, 0.35);
                    ga_sops.run_through();
                },
                Behavior::Coat => {
                    println!("\nStarting Coating GA Experiment...\n");
                    let particle_sizes: Vec<(u16,u16,u16)> = size_strings.map(|c| (c[0],c[1],c[2])).collect::<Vec<(u16,u16,u16)>>();
                    let mut ga_sops = CoatGA::init_ga(args.population, args.max_generations, args.elitist_count, args.mutation_rate, args.granularity, true, particle_sizes, args.seeds, 0.65, 0.35);
                    ga_sops.run_through();
                },
            }
        },
        ref other_experiment => {
            println!("Snapshots: {:?}", &args.snaps);

            assert_eq!(&args.path.is_some(), &true);

            let path = &args.path.clone().unwrap();

            assert_eq!(&path.is_file(), &true);

            println!("Genome file path: {}", &path.display());
            // Read Genome file content and pre-process it
            let contents = fs::read_to_string(&path)
                .expect("Should have been able to read the file");
            let mut striped_content = contents.replace("[", "").replace("]", "").replace(" ", "");
            striped_content.pop();

            /*
             * Perform a standalone evaluation runs of given Genomes for respective behaviours
             * Total Runs = #. of Particle Sizes x #. of Seeds
             */
            match &other_experiment {
                Experiment::GM => {
                    let all_entries: Vec<u8> = striped_content.split(',').filter_map(|x| x.parse::<u8>().ok()).collect();
        
                    match &args.behavior {
                        Behavior::Agg => {
                            println!("\nStarting Aggregation Single Genome Trial...\n");
                            // Construct the genome in required dimension
                            let mut genome: [[[u8; 4]; 3]; 4] = [[[0; 4]; 3]; 4];
                            let mut idx = 0;
                            for n in 0_u8..4 {
                                for j in 0_u8..3 {
                                    for i in 0_u8..4 {
                                        genome[n as usize][j as usize][i as usize] = all_entries[idx];
                                        idx += 1;
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            /*
                             * Sample snippet for computing variance and mean stats of the trials
                            // let scores: Vec<u32> => (for each particle sizes and seed value)
                            // let mean = scores.clone().into_iter().fold(0, |sum, score| sum + score) / 1;
                            // let variance = scores
                            //     .into_iter()
                            //     .fold(0, |sum, score| sum + (score as i32 - mean as i32).pow(2))
                            //     / 1;
                            // println!("N^3 Mean:{}", mean);
                            // println!("N^3 Variance:{}", variance);
                            */

                            // Run the trials in parallel
                            let trials = args.seeds.len();
                            let seeds = args.seeds.clone();

                            let trials_vec: Vec<((u16,u16),u64)> = particle_sizes.clone()
                                .into_iter()
                                .zip(seeds)
                                .flat_map(|v| std::iter::repeat(v).take(trials.into()))
                                .collect();

                            let fitness_tot: f64 = trials_vec.clone()
                            .into_par_iter()
                            .map(|trial| {
                                /*
                                     * Single Evaluation run of the Genome
                                     */
                                    let mut sops_trial = SOPSEnvironment::init_sops_env(&genome, trial.0.0, trial.0.1, trial.1, args.granularity);
                                    sops_trial.print_grid();
                                    let edge_cnt: u32 = sops_trial.evaluate_fitness();
                                    println!("Edge Count: {}", edge_cnt);
                                    println!("Max Fitness: {}", sops_trial.get_max_fitness());
                                    println!("Starting Fitness: {}", edge_cnt as f32/ sops_trial.get_max_fitness() as f32);
                                    let now = Instant::now();
                                    let edge_cnt: u32 = sops_trial.simulate(true);
                                    let elapsed = now.elapsed().as_secs();
                                    sops_trial.print_grid();
                                    println!("Edge Count: {}", edge_cnt);
                                    let t_fitness = edge_cnt as f64/ sops_trial.get_max_fitness() as f64;
                                    println!("Fitness: {}", &t_fitness);
                                    println!("Trial Elapsed Time: {:.2?}s", elapsed);
                                    t_fitness
                            })
                            .sum();

                            println!("Total Fitness: {}", &fitness_tot);
                        },
                        Behavior::Sep => {
                            println!("\nStarting Separation Single Genome Trial...\n");
                            // Construct the genome in required dimension
                            let mut genome: [[[u8; 10]; 6]; 10] = [[[0; 10]; 6]; 10];
                            let mut idx = 0;
                            for n in 0_u8..10 {
                                for j in 0_u8..6 {
                                    for i in 0_u8..10 {
                                        genome[n as usize][j as usize][i as usize] = all_entries[idx];
                                        idx += 1;
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Run the trials in parallel
                            let trials = args.seeds.len();
                            let seeds = args.seeds.clone();

                            let trials_vec: Vec<((u16,u16),u64)> = particle_sizes.clone()
                                .into_iter()
                                .zip(seeds)
                                .flat_map(|v| std::iter::repeat(v).take(trials.into()))
                                .collect();

                            let fitness_tot: f32 = trials_vec.clone()
                            .into_par_iter()
                            .map(|trial| {
                                /*
                                 * Single Evaluation run of the Genome
                                 */
                                let mut sops_trial = SOPSegEnvironment::init_sops_env(&genome,trial.0.0, trial.0.1, trial.1, args.granularity, 0.65, 0.35);
                                sops_trial.print_grid();
                                let fitness: f32 = sops_trial.evaluate_fitness();
                                println!("Starting Fitness: {}", fitness);
                                let now = Instant::now();
                                let t_fitness: f32 = sops_trial.simulate(true);
                                let elapsed = now.elapsed().as_secs();
                                sops_trial.print_grid();
                                println!("Fitness: {}", &t_fitness);
                                println!("Trial Elapsed Time: {:.2?}s", elapsed);
                                t_fitness
                            })
                            .sum();
    
                            println!("Total Fitness: {}", &fitness_tot);
                        },
                        Behavior::Coat => {
                            println!("\nStarting Coating Single Genome Trial...\n");
                            // Construct the genome in required dimension
                            let mut genome: [[[u8; 11]; 7]; 11] = [[[0; 11]; 7]; 11];
                            let mut idx = 0;
                            for n in 0_u8..11 {
                                for j in 0_u8..7 {
                                    for i in 0_u8..11 {
                                        genome[n as usize][j as usize][i as usize] = all_entries[idx];
                                        idx += 1;
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Run the trials in parallel
                            let trials = args.seeds.len();
                            let seeds = args.seeds.clone();

                            let particle_sizes: Vec<(u16,u16,u16)> = size_strings.map(|c| (c[0],c[1],c[2])).collect::<Vec<(u16,u16,u16)>>();

                            let trials_vec: Vec<((u16,u16,u16),u64)> = particle_sizes.clone()
                                .into_iter()
                                .zip(seeds)
                                .flat_map(|v| std::iter::repeat(v).take(trials.into()))
                                .collect();

                            let fitness_tot: f32 = trials_vec.clone()
                            .into_par_iter()
                            .map(|trial| {
                                /*
                                 * Single Evaluation run of the Genome
                                 */
                                let mut sops_trial = SOPSCoatEnvironment::init_sops_env(&genome,trial.0.0, trial.0.1, trial.0.2, trial.1, args.granularity, 0.65, 0.35);
                                sops_trial.print_grid();
                                let fitness: f32 = sops_trial.evaluate_fitness();
                                println!("Starting Fitness: {}", fitness);
                                let now = Instant::now();
                                let t_fitness: f32 = sops_trial.simulate(true);
                                let elapsed = now.elapsed().as_secs();
                                sops_trial.print_grid();
                                println!("Fitness: {}", &t_fitness);
                                println!("Trial Elapsed Time: {:.2?}s", elapsed);
                                t_fitness
                            })
                            .sum();
    
                            println!("Total Fitness: {}", &fitness_tot);
                        },
                    }
                },
                Experiment::TH => {
                    let all_entries: Vec<f32> = striped_content.split(',').map(|x| x.parse::<f32>().unwrap()).collect();
        
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
                        },
                        Behavior::Sep => {
                            let mut genome: [[[f32; 6]; 7]; 7] = [[[0.0; 6]; 7]; 7];
                            let mut idx = 0;
                            for n in 0_u8..7 {
                                for j in 0_u8..7 {
                                    for i in 0_u8..6 {
                                        // if i+j <= n {
                                            genome[n as usize][j as usize][i as usize] = all_entries[idx];
                                            idx += 1;
                                        // }
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        },
                        Behavior::Coat => {
                            let mut genome: [[[f32; 6]; 7]; 7] = [[[0.0; 6]; 7]; 7];
                            let mut idx = 0;
                            for n in 0_u8..7 {
                                for j in 0_u8..7 {
                                    for i in 0_u8..6 {
                                        // if i+j <= n {
                                            genome[n as usize][j as usize][i as usize] = all_entries[idx];
                                            idx += 1;
                                        // }
                                    }
                                }
                            }

                            println!("Read Genome:\n{:?}", genome);

                            // Need to create a new class that takes the Genome's float values and operates on them
                            todo!()
                        },
                    }
                },
                _ => {},
            }
        },
    }

    print_redirect.into_inner();
}
