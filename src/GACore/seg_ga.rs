use crate::SOPSCore::segregation::SOPSegEnvironment;

use super::SegGenome;
use rand::{distributions::Bernoulli, distributions::Uniform, rngs, Rng};
use rayon::prelude::*;
use std::collections::HashMap;
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
    trial_seeds: Vec<u64>
}

impl SegGA {

    const GENOME_LEN: u16 = 10 * 6 * 10;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn genome_init_rng(granularity: u8) -> Uniform<u8> {
        Uniform::new_inclusive(1, granularity)
    }

    #[inline]
    fn mut_frng() -> fastrand::Rng {
        fastrand::Rng::new()
    }

    // fn mut_val(&self) -> Normal<f64> {
    //     Normal::new(self.mut_mu, self.mut_sd).unwrap()
    // }

    fn cross_pnt() -> Uniform<u16> {
        Uniform::new_inclusive(0, SegGA::GENOME_LEN-1)
    }

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
        trial_seeds: Vec<u64>
    ) -> Self {

        println!("Weights: 0.65 x Total + 0.35 x Avg. Same Clr");
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
            trial_seeds
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
                        .clamp(1, self.granularity.into());
                    }
                }
            }
        }
        new_genome
    }

    /*
     * Implements a simple single-point crossover operator with crossover point choosen at random in genome vector
     *  */
    fn generate_offspring(&self, parent1: &[[[u8; 10]; 6]; 10], parent2: &[[[u8; 10]; 6]; 10]) -> [[[u8; 10]; 6]; 10] {
        let mut new_genome: [[[u8; 10]; 6]; 10] = [[[0_u8; 10]; 6]; 10];
        let cross_pnt = SegGA::rng().sample(&SegGA::cross_pnt());
        let mut cnt = 0;
        for n in 0..10 {
            for i in 0..6 {
                for j in 0..10 {
                    if cnt < cross_pnt {
                        new_genome[n][i][j] = parent1[n][i][j];
                    } else {
                        new_genome[n][i][j] = parent2[n][i][j];
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
    fn generate_new_pop(&mut self) {
        let mut new_pop: Vec<SegGenome> = vec![];
        let mut selected_g: Vec<[[[u8; 10]; 6]; 10]> = vec![];
        let mut rank_wheel: Vec<usize> = vec![];
        //sort the genomes in population by fitness value
        self.population.sort_unstable_by(|genome_a, genome_b| {
            genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
        });

        //print genomes for analysis
        let best_genome = self.population[0];
        println!("Best Genome -> {best_genome:.5?}");

        for idx in 1..self.population.len() {
            println!("{y:.5?}", y = self.population[idx].fitness);
        }
        
        //bifercate genomes
        for (index, genome) in self.population.iter().enumerate() {
            if index < self.elitist_cnt as usize {
                //separate out the elitist and directly pass them to next gen
                new_pop.push(*genome);
            }
            let genome_rank = self.population.len() - index;
            //create rank wheel for selection
            for _ in 0..genome_rank {
                rank_wheel.push(index);
            }
        }
        
        //perform selection and then (if perform_cross flag is set) single-point crossover
        let rank_wheel_rng = Uniform::new(0, rank_wheel.len());
        for _ in 0..(self.population.len() - self.elitist_cnt as usize) {
            let mut wheel_idx = SegGA::rng().sample(&rank_wheel_rng);
            let p_genome_idx1 = rank_wheel[wheel_idx];
            if self.perform_cross {
                wheel_idx = SegGA::rng().sample(&rank_wheel_rng);
                let p_genome_idx2 = rank_wheel[wheel_idx];
                selected_g.push(self.generate_offspring(
                    &self.population[p_genome_idx1].string,
                    &self.population[p_genome_idx2].string,
                ));
            } else {
                selected_g.push(self.population[p_genome_idx1].string); // added
            }
        }

        //perform mutation
        for idx in 0..selected_g.len() {
            let genome = selected_g[idx];
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
    fn step_through(&mut self, gen: u16) {
        let trials = self.trial_seeds.len();
        let seeds = self.trial_seeds.clone();
        let granularity = self.granularity.clone();

        let trials_vec: Vec<((u16,u16),u64)> = self
            .sizes.clone()
            .into_iter()
            .zip(seeds)
            .flat_map(|v| std::iter::repeat(v).take(trials.into()))
            .collect();

        // TODO: run each genome in a separate compute node
        // TODO: use RefCell or lazy static to make the whole check and update into a single loop.
        let mut genome_fitnesses = vec![-1.0; self.population.len()];

        // check if the cache has the genome's fitness calculated
        self.population
            .iter()
            .enumerate()
            .for_each(|(idx, genome)| {
                let genome_s = genome.string.clone();
                match self.genome_cache.get(&genome_s) {
                    Some(fitness) => {
                        genome_fitnesses.insert(idx, *fitness);
                        return;
                    }
                    None => return,
                }
            });

        // update the genome if the value exists in the cache
        self.population
            .iter_mut()
            .enumerate()
            .for_each(|(idx, genome)| {
                if genome_fitnesses[idx] > -1.0 {
                    genome.fitness = genome_fitnesses[idx];
                }
            });

        self.population.par_iter_mut().for_each(|genome| {
            // bypass if genome has already fitness value calculated
            let genome_s = genome.string.clone();
            if gen > 0 && genome.fitness > 0.0 {
                return;
            }

            // Calculate the fitness for 'n' number of trials
            let fitness_tot: f64 = trials_vec.clone()
                .into_par_iter()
                .map(|trial| {
                    let mut genome_env = SOPSegEnvironment::init_sops_env(&genome_s, trial.0.0, trial.0.1, trial.1.into(), granularity);
                    let g_fitness = genome_env.simulate(false);
                    // Add normalization of the fitness value based on optimal fitness value for a particular cohort size
                    // let max_fitness = SOPSEnvironment::aggregated_fitness(particle_cnt as u16);
                    // let g_fitness = 1; // added
                    // g_fitness as f64 / (genome_env.get_max_fitness() as f64)
                    g_fitness as f64
                })
                .sum();

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
        for idx in 0..self.population.len() {
            let genome_s = self.population[idx].string.clone();
            let genome_f = self.population[idx].fitness.clone();
            self.genome_cache.insert(genome_s, genome_f);
        }

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
        let mut pop_dist: Vec<f64> = vec![];
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = self.population[i];
                let genome2 = self.population[j];
                let mut dis_sum: f64 = 0.0;
                for n in 0..10 {
                    for i in 0..6 {
                        for j in 0..10 {
                            let genome1_prob = genome1.string[n][i][j] as f64 / (self.granularity as f64);
                            let genome2_prob = genome2.string[n][i][j] as f64 / (self.granularity as f64);
                            let dis = (genome1_prob - genome2_prob).abs();
                            dis_sum += dis.powf(2.0);
                        }
                    }
                }
                // pop_dist.push(dis_sum.sqrt());
                pop_dist.push(dis_sum);
            }
        }
        let pop_diversity: f64 = pop_dist.iter().sum();
        println!(
            "Population diversity -> {}",
            pop_diversity / (pop_dist.len() as f64)
        );
        //generate new population
        self.generate_new_pop();
    }

    /*
     * The main loop of the GA which runs the full scale GA steps untill stopping criterion (ie. MAX Generations)
     * is reached
     *  */
    pub fn run_through(&mut self) {

        // Run the GA for given #. of Generations
        for gen in 0..self.max_gen {
            println!("Starting Gen:{}", gen);
            let now = Instant::now();
            self.step_through(gen);
            let elapsed = now.elapsed().as_secs();
            println!("Elapsed Time: {:.2?}s", elapsed);
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