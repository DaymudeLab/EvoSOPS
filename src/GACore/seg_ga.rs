use crate::SOPSCore::segregation::SOPSegEnvironment;

use super::SegGenome;
use super::DiversThresh;
use rand::{distributions::Bernoulli, distributions::Uniform, rngs, Rng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::io::Write;
use std::fs::File;
use std::time::Instant;
use std::usize;

/*
 * Main GA class for Separation behavior
 * Provides basic 3 operators of the GAs and a step by step (1 step = 1 generation)
 * population generator for each step
 *  */
pub struct SegGA {
    max_gen: u16,
    elitist_cnt: u16,
    population: Vec<SegGenome>,
    mut_rate: f64,
    granularity: u8,
    genome_cache: HashMap<[[[u8; 10]; 6]; 10], f64>,
    perform_cross: bool,
    sizes: Vec<(u16,u16)>,
    trial_seeds: Vec<u64>,
    div_state: DiversThresh,
    max_div: u32,
    w1: f32,
    w2: f32,
    random_seed: u32
}

impl SegGA {

    const GENOME_LEN: u16 = 10 * 6 * 10;
    const BUFFER_LEN: usize = 10;
    const UPPER_T: f32 = 0.3;
    const LOWER_T: f32 = 0.08;
    
    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn genome_init_rng(granularity: u8) -> Uniform<u8> {
        Uniform::new_inclusive(0, granularity)
    }

    #[inline]
    fn genome_rng(population_size: u16) -> Uniform<u16> {
        Uniform::new(0, population_size)
    }

    #[inline]
    fn mut_frng() -> fastrand::Rng {
        fastrand::Rng::new()
    }

    // fn mut_val(&self) -> Normal<f64> {
    //     Normal::new(self.mut_mu, self.mut_sd).unwrap()
    // }
    #[inline]
    fn cross_pnt() -> Uniform<u16> {
        Uniform::new_inclusive(0, SegGA::GENOME_LEN-1)
    }

    #[inline]
    fn mut_sign() -> Bernoulli {
        Bernoulli::new(0.3).unwrap()
    }

    /*
     * Initialize GA with given parameters and a random set of genome vectors
     *  */
    #[inline]
    pub fn init_ga(
        population_size: u16,
        max_gen: u16,
        elitist_cnt: u16,
        mut_rate: f64,
        granularity: u8,
        perform_cross: bool,
        sizes: Vec<(u16, u16)>,
        trial_seeds: Vec<u64>,
        w1: f32,
        w2: f32,
        random_seed: u32,
    ) -> Self {

        println!("Weights: {} x Total Edges + {} x Avg. Same Clr Edges", w1, w2);
        println!("Thresholds: UPPER: {}, LOWER {}", SegGA::UPPER_T, SegGA::LOWER_T);
        let mut starting_pop: Vec<SegGenome> = vec![];

        for _ in 0..population_size {
            //init genome
            let mut genome: [[[u8; 10]; 6]; 10] = [[[0_u8; 10]; 6]; 10];
            for n in 0_u8..10 {
                for j in 0_u8..6 {
                    for i in 0_u8..10 {
                        genome[n as usize][j as usize][i as usize] = SegGA::rng().sample(SegGA::genome_init_rng(granularity))
                    }
                }
            }
            starting_pop.push(SegGenome {
                string: (genome),
                fitness: (0.0),
            });
        }

        let genome_cache: HashMap<[[[u8; 10]; 6]; 10], f64> = HashMap::new();

        SegGA {
            max_gen,
            elitist_cnt,
            population: starting_pop,
            mut_rate,
            granularity,
            genome_cache,
            perform_cross,
            sizes,
            trial_seeds,
            div_state: DiversThresh::INIT,
            max_div: ((granularity-1) as u32)*(SegGA::GENOME_LEN as u32),
            w1,
            w2,
            random_seed
        }
    }

    // mutate genome based on set mutation rate for every gene of the genome
    fn mutate_genome(&self, genome: &[[[u8; 10]; 6]; 10]) -> [[[u8; 10]; 6]; 10] {
        let mut new_genome = genome.clone();
        for n in 0..10 {
            for i in 0..6 {
                for j in 0..10 {
                    let smpl = SegGA::mut_frng().u64(1_u64..=10000);
                    if smpl as f64 <= (self.mut_rate * 10000.0) {
                        // a random + or - mutation operation on each gene
                        let per_dir = SegGA::rng().sample(&SegGA::mut_sign());
                        new_genome[n][i][j] = (if per_dir {
                            genome[n][i][j] + 1
                        } else if genome[n][i][j] == 0 {
                            0
                        } else {
                            genome[n][i][j] - 1
                        })
                        .clamp(0, self.granularity.into());
                    }
                }
            }
        }
        new_genome
    }

    /*
     * Implements a simple single-point crossover operator with crossover point choosen at random in genome vector
     *  */
    // fn generate_offspring(&self, parent1: &[[[u8; 10]; 6]; 10], parent2: &[[[u8; 10]; 6]; 10]) -> [[[u8; 10]; 6]; 10] {
    //     let mut new_genome: [[[u8; 10]; 6]; 10] = [[[0_u8; 10]; 6]; 10];
    //     let cross_pnt = SegGA::rng().sample(&SegGA::cross_pnt());
    //     let mut cnt = 0;
    //     for n in 0..10 {
    //         for i in 0..6 {
    //             for j in 0..10 {
    //                 if cnt < cross_pnt {
    //                     new_genome[n][i][j] = parent1[n][i][j];
    //                 } else {
    //                     new_genome[n][i][j] = parent2[n][i][j];
    //                 }
    //                 cnt += 1; 
    //             }
    //         }
    //     }
    //     new_genome
    // }

    /*
     * Implements a simple two-point crossover operator with crossover point choosen at random in genome vector
     *  */
    fn generate_offspring(&self, parent1: &[[[u8; 10]; 6]; 10], parent2: &[[[u8; 10]; 6]; 10]) -> [[[u8; 10]; 6]; 10] {
        let mut new_genome: [[[u8; 10]; 6]; 10] = [[[0_u8; 10]; 6]; 10];
        let cross_pnt_1 = SegGA::rng().sample(&SegGA::cross_pnt());
        let cross_pnt_2 = SegGA::rng().sample(&SegGA::cross_pnt());
        let lower_cross_pnt = if cross_pnt_1 <= cross_pnt_2 {cross_pnt_1} else {cross_pnt_2};
        let higher_cross_pnt = if cross_pnt_1 > cross_pnt_2 {cross_pnt_1} else {cross_pnt_2};

        let mut cnt = 0;
        for n in 0..10 {
            for i in 0..6 {
                for j in 0..10 {
                    if cnt < lower_cross_pnt {
                        new_genome[n][i][j] = parent1[n][i][j];
                    } else if cnt > lower_cross_pnt && cnt < higher_cross_pnt {
                        new_genome[n][i][j] = parent2[n][i][j];
                    } else {
                        new_genome[n][i][j] = parent1[n][i][j];
                    }
                    cnt += 1;
                }
            }
        }
        new_genome
    }

    /*
     * Performs the 3 operations (in sequence 1. selection, 2. crossover, 3. mutation) 
     * on the existing populations to generate new population
     *  */
    // fn generate_new_pop(&mut self) {
    //     let mut new_pop: Vec<SegGenome> = vec![];
    //     let mut selected_g: Vec<[[[u8; 10]; 6]; 10]> = vec![];
    //     let mut rank_wheel: Vec<usize> = vec![];
    //     //sort the genomes in population by fitness value
    //     self.population.sort_unstable_by(|genome_a, genome_b| {
    //         genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
    //     });

    //     //print genomes for analysis
    //     let best_genome = self.population[0];
    //     println!("Best Genome -> {best_genome:.5?}");

    //     for idx in 1..self.population.len() {
    //         println!("{y:.5?}", y = self.population[idx].fitness);
    //     }
        
    //     //bifercate genomes
    //     for (index, genome) in self.population.iter().enumerate() {
    //         if index < self.elitist_cnt as usize {
    //             //separate out the elitist and directly pass them to next gen
    //             new_pop.push(*genome);
    //         }
    //         let genome_rank = self.population.len() - index;
    //         //create rank wheel for selection
    //         for _ in 0..genome_rank {
    //             rank_wheel.push(index);
    //         }
    //     }
    //     //perform selection and then (if perform_cross flag is set) single-point crossover
    //     let rank_wheel_rng = Uniform::new(0, rank_wheel.len());
    //     for _ in 0..(self.population.len() - self.elitist_cnt as usize) {
    //         let mut wheel_idx = SegGA::rng().sample(&rank_wheel_rng);
    //         let p_genome_idx1 = rank_wheel[wheel_idx];
    //         if self.perform_cross {
    //             wheel_idx = SegGA::rng().sample(&rank_wheel_rng);
    //             let p_genome_idx2 = rank_wheel[wheel_idx];
    //             selected_g.push(self.generate_offspring(
    //                 &self.population[p_genome_idx1].string,
    //                 &self.population[p_genome_idx2].string,
    //             ));
    //         } else {
    //             selected_g.push(self.population[p_genome_idx1].string); // added
    //         }
    //     }

    //     //perform mutation
    //     for idx in 0..selected_g.len() {
    //         let genome = selected_g[idx];
    //         // println!("Genome:{} mutations", idx);
    //         let mutated_g = self.mutate_genome(&genome);
    //         new_pop.push(SegGenome {
    //             string: mutated_g,
    //             fitness: 0.0,
    //         });
    //     }
    //     self.population = new_pop;
    // }

    /*
     * Performs the 3 operations (in sequence 1. selection, 2. crossover, 3. mutation) 
     * on the existing populations to generate new population
     *  */
     fn generate_new_pop(&mut self) {
        let mut new_pop: Vec<SegGenome> = vec![];
        let mut selected_g: Vec<[[[u8; 10]; 6]; 10]> = vec![];
        let mut crossed_g: Vec<[[[u8; 10]; 6]; 10]> = vec![];
        let population_size = self.population.len() as u16;
        //sort the genomes in population by fitness value
        // self.population.sort_unstable_by(|genome_a, genome_b| {
        //     genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
        // });

        //print genomes for analysis
        let best_genome = self.population.iter().max_by(|&g1, &g2| g1.fitness.partial_cmp(&g2.fitness).unwrap()).unwrap();
        println!("Best Genome -> {best_genome:.5?}");

        // Write all the genomic data to a file
        {
            let mut buff: Vec<u8> = Vec::new();
            for genome in &self.population {
                for n in 0..10 {
                    for i in 0..6 {
                        for j in 0..10 {
                            buff.push(genome.string[n][i][j]);
                        }
                    }
                }
                buff.extend(genome.fitness.to_be_bytes());
            }

            let mut file = File::options().create(true).append(true).open(format!("./output/genomic_data_Sep_{}.log", self.random_seed)).expect("Failed to create genomic data file!");
            file.write_all(&buff).expect("Failed to append to the genomic data file!");
        }
        
        // for idx in 1..self.population.len() {
        //     println!("{y:.5?}", y = self.population[idx].fitness);
        // }
        
        //perform tournament selection
        for _ in 0..(population_size) {
            let genome_idx_1 = SegGA::rng().sample(&SegGA::genome_rng(population_size));
            let mut genome_idx_2;
            loop {
                genome_idx_2 = SegGA::rng().sample(&SegGA::genome_rng(population_size));
                if genome_idx_1 != genome_idx_2 {
                    break;
                }
            }
            let genome_1 = self.population[genome_idx_1 as usize];
            let genome_2 = self.population[genome_idx_2 as usize];
            if genome_1.fitness > genome_2.fitness {
                selected_g.push(genome_1.string);
            } else {
                selected_g.push(genome_2.string);
            }
        }
        
        //perform 2-point crossover
        for _ in 0..(population_size) {
            let genome_idx_1 = SegGA::rng().sample(&SegGA::genome_rng(population_size));
            let mut genome_idx_2;
            loop {
                genome_idx_2 = SegGA::rng().sample(&SegGA::genome_rng(population_size));
                if genome_idx_1 != genome_idx_2 {
                    break;
                }
            }
            let genome_1 = selected_g[genome_idx_1 as usize];
            let genome_2 = selected_g[genome_idx_2 as usize];
            crossed_g.push(self.generate_offspring(&genome_1,&genome_2));
        }

        //perform mutation
        for idx in 0..(population_size) {
            let genome = crossed_g[idx as usize];
            // println!("Genome:{} mutations", idx);
            let mutated_g = self.mutate_genome(&genome);
            new_pop.push(SegGenome {
                string: mutated_g,
                fitness: 0.0,
            });
        }
        self.population = new_pop;
    }

    // A single step of GA ie. generation, where following happens in sequence
    // 1. calculate new population's fitness values
    // 2. Save each genome's fitness value based on mean fitness for 'n' eval trials
    // 3. Generate new population based on these fitness values
    fn step_through(&mut self, gen: u16) -> f32 {
        let trials = self.trial_seeds.len();
        let seeds = self.trial_seeds.clone();
        let granularity = self.granularity.clone();
        let w1 = self.w1.clone();
        let w2 = self.w2.clone();

        println!("Mutation Rate:{}", self.mut_rate);
        println!("Diversity State:{:?}", self.div_state);

        // let trials_vec: Vec<((u16,u16),u64)> = self
        //     .sizes.clone()
        //     .into_iter()
        //     .zip(seeds)
        //     .flat_map(|v| std::iter::repeat(v).take(trials.into()))
        //     .collect();

        let mut trials_vec: Vec<((u16,u16),u64)> = Vec::new();

        self.sizes.iter().for_each(|size| {
            self.trial_seeds.iter().for_each(|seed| {
                trials_vec.push(((size.0,size.1),*seed));
            });
        });

        // trials_vec.iter().for_each(|t| println!("{:?}", t));
        // TODO: run each genome in a separate compute node
        // TODO: use RefCell or lazy static to make the whole check and update into a single loop.
        // let mut genome_fitnesses = vec![-1.0; self.population.len()];

        // check if the cache has the genome's fitness calculated
        // self.population
        //     .iter()
        //     .enumerate()
        //     .for_each(|(idx, genome)| {
        //         let genome_s = genome.string.clone();
        //         match self.genome_cache.get(&genome_s) {
        //             Some(fitness) => {
        //                 println!("Cache Hit!");
        //                 genome_fitnesses.insert(idx, *fitness);
        //                 return;
        //             }
        //             None => return,
        //         }
        //     });

        // update the genome if the value exists in the cache
        // self.population
        //     .iter_mut()
        //     .enumerate()
        //     .for_each(|(idx, genome)| {
        //         if genome_fitnesses[idx] > -1.0 {
        //             genome.fitness = genome_fitnesses[idx];
        //         }
        //     });

        self.population.par_iter_mut().for_each(|genome| {
            // bypass if genome has already fitness value calculated
            let genome_s = genome.string.clone();
            if gen > 0 && genome.fitness > 0.0 {
                return;
            }

            // println!();
            // let now = Instant::now();
            // Calculate the fitness for 'n' number of trials
            let fitness_tot: f64 = trials_vec.clone()
                .into_par_iter()
                .map(|trial| {
                    // let now = Instant::now();
                    let mut genome_env = SOPSegEnvironment::init_sops_env(&genome_s, trial.0.0, trial.0.1, trial.1.into(), granularity, w1, w2);
                    let g_fitness = genome_env.simulate(false);
                    // Add normalization of the fitness value based on optimal fitness value for a particular cohort size
                    // let max_fitness = SOPSEnvironment::aggregated_fitness(particle_cnt as u16);
                    // let g_fitness = 1; // added
                    // g_fitness as f64 / (genome_env.get_max_fitness() as f64)
                    // let elapsed = now.elapsed().as_secs();
                    // print!("{:.2?} ", elapsed);
                    g_fitness as f64
                })
                .sum();

            // let elapsed = now.elapsed().as_secs();
            // println!("Genome Elapsed Time: {:.2?}s", elapsed);
            /* Snippet to calculate Median fitness value of the 'n' trials
            // let mut sorted_fitness_eval: Vec<f64> = Vec::new();
            // fitness_trials.collect_into_vec(&mut sorted_fitness_eval);
            // sorted_fitness_eval.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // println!("Trials: {y:?}",y = sorted_fitness_eval);
            // println!("Mid: {y}",y=((trials / 2) as usize));
            // genome.fitness = sorted_fitness_eval[((trials / 2) as usize)];
            */

            let fitness_val = fitness_tot / (trials_vec.len() as f64) as f64;
            genome.fitness = fitness_val;
        });

        // populate the cache
        // for idx in 0..self.population.len() {
        //     let genome_s = self.population[idx].string.clone();
        //     let genome_f = self.population[idx].fitness.clone();
        //     self.genome_cache.insert(genome_s, genome_f);
        // }

        //avg.fitness of population
        let fit_sum = self
            .population
            .iter()
            .fold(0.0, |sum, genome| sum + genome.fitness);
        println!(
            "Avg. Fitness -> {}",
            fit_sum / (self.population.len() as f64)
        );

        // calculate population diversity
        // based on simple component wise euclidean distance squared*
        // of the genome vectors
        let mut pop_dist: Vec<f32> = vec![];
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = self.population[i];
                let genome2 = self.population[j];
                let mut dis_sum: u16 = 0;
                for n in 0..10 {
                    for i in 0..6 {
                        for j in 0..10 {
                            let dis = (genome1.string[n][i][j]).abs_diff(genome2.string[n][i][j]);
                            dis_sum += dis as u16;
                            // let genome1_prob = genome1.string[n][i][j] as f32 / (self.granularity as f32);
                            // let genome2_prob = genome2.string[n][i][j] as f32 / (self.granularity as f32);
                            // let dis = (genome1_prob - genome2_prob).abs();
                            // dis_sum += dis.powf(2.0); // why square ??
                        }
                    }
                }
                // pop_dist.push(dis_sum.sqrt());
                pop_dist.push(dis_sum.into());
            }
        }
        let pop_diversity: f32 = pop_dist.iter().sum();
        let avg_pop_diversity: f32 = pop_diversity / (pop_dist.len() as f32);
        println!(
            "Population diversity -> {}",
            avg_pop_diversity / (self.max_div as f32)
        );
        //generate new population
        self.generate_new_pop();
        avg_pop_diversity
    }

    fn increase_mut(&mut self) {
        self.mut_rate = self.mut_rate * 10.0;
    }

    fn lower_mut(&mut self) {
        self.mut_rate = self.mut_rate / 10.0;
    }

    /*
     * The main loop of the GA which runs the full scale GA steps untill stopping criterion (ie. MAX Generations)
     * is reached
     *  */
    pub fn run_through(&mut self) {
        let mut diversity_q: VecDeque<f32> = VecDeque::with_capacity(SegGA::BUFFER_LEN);
        // Run the GA for given #. of Generations
        for gen in 0..self.max_gen {
            println!("Starting Gen:{}", gen);
            let now = Instant::now();
            if diversity_q.len() == SegGA::BUFFER_LEN { diversity_q.pop_front(); }
            diversity_q.push_back(self.step_through(gen));
            let avg_div: f32 = diversity_q.iter().sum::<f32>() / (diversity_q.len() as f32);
            let norm_avg_div = avg_div / (self.max_div as f32);
            // println!("Avg. Population diversity for last {} gen -> {}", SegGA::BUFFER_LEN, avg_div);
            println!("Avg. Population diversity for last {} gen -> {}", diversity_q.len(), norm_avg_div);
            match self.div_state {
                DiversThresh::INIT => {
                    if norm_avg_div <= SegGA::LOWER_T {
                        self.increase_mut();
                        self.div_state = DiversThresh::LOWER_HIT;
                    }
                },
                DiversThresh::LOWER_HIT => {
                    if norm_avg_div >= SegGA::UPPER_T {
                        self.lower_mut();
                        self.div_state = DiversThresh::INIT;
                    }
                }
            }
            let elapsed = now.elapsed().as_secs();
            println!("Generation Elapsed Time: {:.2?}s", elapsed);
        }
        /*
         * Snippet to evaluate the final best genome evolved at the end of GA execution
         * TODO: Accept a parameter to run this snippet ?? Or save the best genomes to files if need be ?
        // let best_genome = self.population[0];
        // let mut best_genome_env = SOPSEnvironment::static_init(&best_genome.string);
        // best_genome_env.print_grid();
        // let g_fitness = best_genome_env.simulate();
        // best_genome_env.print_grid();
        // println!("Best genome's fitness is {}", g_fitness);
        // println!("{best_genome:?}");
         */
    }
}