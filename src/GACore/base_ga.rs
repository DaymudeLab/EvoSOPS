use crate::SOPSCore::SOPSEnvironment;

use super::Genome;
use rand::{distributions::Bernoulli, distributions::Uniform, rngs, Rng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::convert::TryInto;
use std::time::Instant;
use std::usize;

pub struct GeneticAlgo {
    max_gen: u16,
    elitist_cnt: u16,
    population: Vec<Genome>,
    mut_rate: f64,
    granularity: u16,
    genome_cache: HashMap<[u16; 6], f64>,
    perform_cross: bool,
    sizes: Vec<(u16,u16)>,
    trial_seeds: Vec<u64>
}

impl GeneticAlgo {
    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn genome_init_rng(granularity: u16) -> Uniform<u16> {
        Uniform::new_inclusive(1, granularity)
    }

    #[inline]
    fn unfrm_100() -> Uniform<u16> {
        Uniform::new_inclusive(1, 100)
    }

    // fn mut_val(&self) -> Normal<f64> {
    //     Normal::new(self.mut_mu, self.mut_sd).unwrap()
    // }

    fn cross_pnt() -> Uniform<u16> {
        Uniform::new_inclusive(0, 5)
    }

    fn mut_sign() -> Bernoulli {
        Bernoulli::new(0.3).unwrap()
    }

    #[inline]
    pub fn init_ga(
        population_size: u16,
        max_gen: u16,
        elitist_cnt: u16,
        mut_rate: f64,
        granularity: u16,
        perform_cross: bool,
        sizes: Vec<(u16, u16)>,
        trial_seeds: Vec<u64>
    ) -> Self {
        let mut starting_pop: Vec<Genome> = vec![];

        for _ in 0..population_size {
            //init genome
            let genome: [u16; 6] = TryInto::try_into(
                (0..6)
                    .map(|_| GeneticAlgo::rng().sample(GeneticAlgo::genome_init_rng(granularity)))
                    .collect::<Vec<u16>>(),
            )
            .unwrap();
            starting_pop.push(Genome {
                string: (genome),
                fitness: (0.0),
            });
        }

        // let mut genome_cache: RefCell<HashMap<[u16; 6], f64>> = RefCell::new(HashMap::new());
        let genome_cache: HashMap<[u16; 6], f64> = HashMap::new();

        GeneticAlgo {
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

    fn mutate_genome(&self, genome: &[u16; 6]) -> [u16; 6] {
        let mut new_genome = genome.clone();
        //mutate genome
        for i in 0..genome.len() {
            let smpl = GeneticAlgo::rng().sample(&GeneticAlgo::unfrm_100());
            if smpl as f64 <= self.mut_rate * 100.0 {
                // let perturb = SOPSEnvironment::rng().sample(self.mut_val());
                let per_dir = GeneticAlgo::rng().sample(&GeneticAlgo::mut_sign());
                new_genome[i] = (if per_dir {
                    genome[i] + 1
                } else if genome[i] == 0 {
                    0
                } else {
                    genome[i] - 1
                })
                .clamp(1, self.granularity);
                // let scale = if per_dir { 0.1 } else { -0.1 };
                // print!("{y:.5?} ",y=scale*genome[i]);
                // new_genome[i] = (genome[i] + scale*genome[i]).clamp(0.0, 1.0);
            }
            // print!("\t");
        }
        // println!("");
        new_genome
    }

    fn generate_offspring(&self, parent1: &[u16; 6], parent2: &[u16; 6]) -> [u16; 6] {
        let mut new_genome: [u16; 6] = [0; 6];
        let cross_pnt = GeneticAlgo::rng().sample(&GeneticAlgo::cross_pnt());
        for i in 0..new_genome.len() {
            if i as u16 <= cross_pnt {
                new_genome[i] = parent1[i];
            } else {
                new_genome[i] = parent2[i];
            }
        }
        new_genome
    }

    fn generate_new_pop(&mut self) {
        let mut new_pop: Vec<Genome> = vec![];
        let mut selected_g: Vec<[u16; 6]> = vec![];
        let mut rank_wheel: Vec<usize> = vec![];
        //sort the genomes in population by fitness value
        self.population.sort_unstable_by(|genome_a, genome_b| {
            genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
        });
        // /*
        //print genomes for analysis
        let best_genome = self.population[0];
        println!("Best Genome -> {best_genome:.5?}");

        for idx in 1..self.population.len() {
            println!("{y:.5?}", y = self.population[idx]);
        }
        // */
        // /*
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
            let mut wheel_idx = GeneticAlgo::rng().sample(&rank_wheel_rng);
            let p_genome_idx1 = rank_wheel[wheel_idx];
            if self.perform_cross {
                wheel_idx = GeneticAlgo::rng().sample(&rank_wheel_rng);
                let p_genome_idx2 = rank_wheel[wheel_idx];
                selected_g.push(self.generate_offspring(
                    &self.population[p_genome_idx1].string,
                    &self.population[p_genome_idx2].string,
                ));
            } else {
                selected_g.push(self.population[p_genome_idx1].string); // added
            }
        }
        // */
        /*
        //dummy pass
        for idx in 0..(self.population.len()) {
            // let mut wheel_idx = SOPSEnvironment::rng().sample(&rank_wheel_rng);
            // let p_genome_idx1 = rank_wheel[wheel_idx];
            // wheel_idx = SOPSEnvironment::rng().sample(&rank_wheel_rng);
            // let p_genome_idx2 = rank_wheel[wheel_idx];
            // selected_g.push(self.generate_offspring(
            //     &self.population[p_genome_idx1].string,
            //     &self.population[p_genome_idx2].string,
            // ));
            selected_g.push(self.population[idx].string); // added
        }
        */

        //perform mutation
        for idx in 0..selected_g.len() {
            let genome = selected_g[idx];
            // println!("Genome:{} mutations", idx);
            let mutated_g = self.mutate_genome(&genome);
            new_pop.push(Genome {
                string: mutated_g,
                fitness: 0.0,
            });
        }
        self.population = new_pop;
    }

    fn step_through(&mut self, gen: u16) {
        //simulate a single step for the ga -> calculate new population's fitness values
        let trials = self.trial_seeds.len();
        let seeds = self.trial_seeds.clone();
        // let trials_vec: Vec<(u16, usize)> = vec![(0,0); (trials*(self.sizes.len() as u16)).into()];

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

            // A specific size calculate the fitness for 'n' number of trials
            let fitness_tot: f64 = trials_vec.clone()
                .into_par_iter()
                .map(|trial| {
                    let mut genome_env = SOPSEnvironment::init_sops_env(&genome_s, trial.0.0, trial.0.1, trial.1.into());
                    let g_fitness = genome_env.simulate(false);
                    // Add normalization of the fitness value based on optimal fitness value for a particular cohort size
                    // let max_fitness = SOPSEnvironment::aggregated_fitness(particle_cnt as u16);
                    // let g_fitness = 1; // added
                    g_fitness as f64 / (genome_env.get_max_edge_cnt() as f64)
                })
                .sum();
            // Choose the median of the returned values
            // let mut sorted_fitness_eval: Vec<f64> = Vec::new();
            // fitness_trials.collect_into_vec(&mut sorted_fitness_eval);
            // sorted_fitness_eval.sort_by(|a, b| a.partial_cmp(b).unwrap());
            //
            // for _ in 0..trials {
            //     // let mut genome_env = SOPSEnvironment::static_init(&genome_s);
            //     // let g_fitness = genome_env.simulate();
            //     let g_fitness: f64 = genome_s.iter().sum(); // added
            //     fitness_t += g_fitness;
            // }
            let fitness_val = fitness_tot / (trials_vec.len() as f64) as f64;
            // println!("Trials: {y:?}",y = sorted_fitness_eval);
            // println!("Mid: {y}",y=((trials / 2) as usize));
            // genome.fitness = sorted_fitness_eval[((trials / 2) as usize)];
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
        //calculate population diversity
        let mut pop_dist: Vec<f64> = vec![];
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = self.population[i];
                let genome2 = self.population[j];
                let genome1_sum = genome1.string.iter().sum::<u16>() as f64;
                let genome2_sum = genome2.string.iter().sum::<u16>() as f64;
                let mut dis_sum: f64 = 0.0;
                for idx in 0..genome1.string.len() {
                    let genome1_prob = genome1.string[idx] as f64 / genome1_sum;
                    let genome2_prob = genome2.string[idx] as f64 / genome2_sum;
                    let dis = (genome1_prob - genome2_prob).abs();
                    dis_sum += dis.powf(2.0);
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

    pub fn run_through(&mut self) {
        // let utc: DateTime<Utc> = Utc::now();
        // let file_path = format!("{ROOT_PATH}exp_{dt}",dt = utc.format("%d_%m_%Y_%H:%M"));
        // let mut file = File::create(file_path)?;
        println!("Max Gen:{y}", y = self.max_gen);
        println!("Trial Count:{y}", y = self.trial_seeds.len());
        println!("mutation Rate:{y}", y = self.mut_rate);
        println!("Granularity:{y}", y = self.granularity);
        println!("Crossover ?:{y}", y = self.perform_cross);
        println!("Population Size:{y}", y = self.population.len());
        for gen in 0..self.max_gen {
            println!("Starting Gen:{}", gen);
            let now = Instant::now();
            self.step_through(gen);
            let elapsed = now.elapsed().as_secs();
            println!("Elapsed Time: {:.2?}", elapsed);
            // std::io::stdout().flush().expect("some error message");
        }
        // let best_genome = self.population[0];
        // let mut best_genome_env = SOPSEnvironment::static_init(&best_genome.string);
        // best_genome_env.print_grid();
        // let g_fitness = best_genome_env.simulate();
        // best_genome_env.print_grid();
        // println!("Best genome's fitness is {}", g_fitness);
        // println!("{best_genome:?}");
    }
}