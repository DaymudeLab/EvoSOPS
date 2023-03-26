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

struct Particle {
    x: u8,
    y: u8,
}
#[derive(Debug, Copy, Clone)]
struct Genome {
    string: [u16; 6],
    fitness: f64,
}

struct SOPSEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [u16; 6],
    sim_duration: u64,
    fitness_val: f64,
    size: usize,
    max_edge_cnt: u64,
    arena_layers: u16,
    particle_layers: u16
}

struct GeneticAlgo {
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

impl SOPSEnvironment {
    const par_density: f64 = 0.5;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn seed_rng(seed: u64) -> rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[inline]
    fn move_nrng() -> WyRand {
        WyRand::new()
    }

    #[inline]
    fn move_frng() -> fastrand::Rng {
        fastrand::Rng::new()
    }

    #[inline]
    fn grid_rng(size: usize) -> Uniform<usize> {
        Uniform::new(0, size)
    }

    #[inline]
    fn unfrm_move() -> Uniform<u64> {
        Uniform::<u64>::new(0, 1000)
    }

    #[inline]
    fn unfrm_dir() -> Uniform<usize> {
        Uniform::new(0, 6)
    }

    #[inline]
    fn directions() -> Vec<(i32, i32)> {
        vec![(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
    }

    #[inline]
    fn aggregated_fitness(particle_cnt: u16) -> u32 {
        return 0;
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    /*
    fn _static_init(genome: &[u16; 6], arena_layers: u16, particle_layers: u16, seed: u64) -> Self {
        // initial starting fitness of given configuration is 154
        let init_grid: [[u8; 18]; 18] = [
            [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
            ],
            [
                0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
            ],
            [
                0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            ],
            [
                1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0,
            ],
            [
                1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
            ],
            [
                0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
            ],
            [
                0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
            ],
            [
                0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
            ],
            [
                1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
            ],
            [
                0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0,
            ],
            [
                1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
            ],
            [
                0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
                1, 1, 0, 1, 1, 0,
            ],
            [
                1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                1, 0, 0, 1, 1, 1,
            ],
            [
                1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1,
                0, 1, 0, 0, 0,
            ],
            [
                0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0,
                1, 0, 1, 0, 0, 0,
            ],
            [
                1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                1, 0, 0, 1, 0,
            ],
            [
                0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                0, 1, 0, 0, 0, 0,
            ],
        ];
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        //init grid and particles
        for i in 0..grid_size {
            for j in 0..grid_size {
                if init_grid[i as usize][j as usize] == 1 {
                    participants.push(Particle {
                        x: i as u8,
                        y: j as u8,
                    });
                }
                grid[i as usize][j  as usize] = init_grid[i as usize][j as usize];
            }
        }

        let particle_cnt = participants.len();

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (particle_cnt as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
        }
    }
    */
    // No need for static init with Seeded Random Generator
    // Use a Random Seed value to get the previous random init behaviour
    fn init_sops_env(genome: &[u16; 6], arena_layers: u16, particle_layers: u16, seed: u64) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let num_particles = 6*particle_layers*(1+particle_layers)/2 + 1;
        let k = 3*particle_layers;
        let agg_edge_cnt: u64 = (k*(k+1)).into();
        let mut grid_rng = SOPSEnvironment::seed_rng(seed);
        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = 2;
                grid[(i+arena_layers+j) as usize][i as usize] = 2;
                j +=1;
            }
        }

        //init grid and particles
        while participants.len() < num_particles.into() {
            let i = grid_rng.sample(&SOPSEnvironment::grid_rng(grid_size));
            let j = grid_rng.sample(&SOPSEnvironment::grid_rng(grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                });
                grid[i][j] = 1;
            }
            
        }

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_edge_cnt: agg_edge_cnt,
            arena_layers,
            particle_layers
        }
    }

    fn print_grid(&self) {
        println!("SOPS grid");
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                print!(" {} ", self.grid[i][j]);
            }
            println!("");
        }
    }

    fn get_neighbors_cnt(&self, i: u8, j: u8) -> u8 {
        let mut cnt = 0;
        for idx in 0..6 {
            let new_i = (i as i32 + SOPSEnvironment::directions()[idx].0) as usize;
            let new_j = (j as i32 + SOPSEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                if self.grid[new_i][new_j] == 1 {
                    cnt += 1;
                }
            }
        }
        cnt
    }

    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let mut particle = &mut self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            if self.grid[new_i][new_j] == 1 || self.grid[new_i][new_j] == 2 {
                return false;
            } else {
                self.grid[particle.x as usize][particle.y as usize] = 0;
                self.grid[new_i][new_j] = 1;
                particle.x = new_i as u8;
                particle.y = new_j as u8;
                return true;
            }
        } else {
            return false;
        }
    }

    fn move_particles(&mut self, cnt: usize) {
        let mut par_moves: Vec<(usize, (i32, i32))> = Vec::new();
        for _ in 0..cnt {
            let par_idx = SOPSEnvironment::rng().sample(&self.unfrm_par());
            let particle: &Particle = &self.participants[par_idx];
            let n_cnt = self.get_neighbors_cnt(particle.x, particle.y) as usize;
            if n_cnt == 6 {
                continue;
            }
            let move_prb: f64 =
                self.phenotype[n_cnt] as f64 / (self.phenotype.iter().sum::<u16>() as f64);
            // if SOPSEnvironment::move_nrng().generate_range(1_u64..=1000)
            //     <= (move_prb * 1000.0) as u64
            // {
            //     let move_dir = SOPSEnvironment::directions()
            //         [SOPSEnvironment::move_nrng().generate_range(1..6)];
            //     par_moves.push((par_idx, move_dir));
            // }
            if SOPSEnvironment::move_frng().u64(1_u64..=1000)
                <= (move_prb * 1000.0) as u64
            {
                let move_dir = SOPSEnvironment::directions()
                    [SOPSEnvironment::rng().sample(&SOPSEnvironment::unfrm_dir())];
                par_moves.push((par_idx, move_dir));
            }
        }

        // Parallel execution
        /*
        let par_moves: Vec<(usize, (i32, i32))> = (0..cnt).into_par_iter().filter_map(|_| {
            let par_idx = SOPSEnvironment::move_frng().usize(0..self.participants.len());
            let particle: &Particle = &self.participants[par_idx];
            let n_cnt = self.get_neighbors_cnt(particle.x, particle.y) as usize;
            if n_cnt == 6 {
                return None;
            }
            let move_prb: f64 =
                self.phenotype[n_cnt] as f64 / (self.phenotype.iter().sum::<u16>() as f64);
            // if SOPSEnvironment::move_nrng().generate_range(1_u64..=1000)
            //     <= (move_prb * 1000.0) as u64
            // {
            //     let move_dir = SOPSEnvironment::directions()
            //         [SOPSEnvironment::move_nrng().generate_range(1..6)];
            //     par_moves.push((par_idx, move_dir));
            // }
            if SOPSEnvironment::move_frng().u64(1_u64..=1000)
                <= (move_prb * 1000.0) as u64
            {
                let move_dir = SOPSEnvironment::directions()
                    [SOPSEnvironment::move_frng().usize(1..6)];
                return Some((par_idx, move_dir));
            }
            return None;
        }).collect();
         */

        for moves in par_moves.iter() {
            self.move_particle_to(moves.0, moves.1);
        }
    }

    fn evaluate_fitness(&self) -> u32 {
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            sum + self.get_neighbors_cnt(particle.x, particle.y) as u32
        });
        edges / 2
    }

    fn simulate(&mut self) -> u32 {
        for _ in 0..self.sim_duration {
            // let now = Instant::now();
            self.move_particles((self.participants.len() as f32 * 0.03) as usize);
            // let elapsed = now.elapsed().as_micros();
            // println!("Step Elapsed Time: {:.2?}", elapsed);
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness as f64;
        fitness
    }
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
                    let g_fitness = genome_env.simulate();
                    // Add normalization of the fitness value based on optimal fitness value for a particular cohort size
                    // let max_fitness = SOPSEnvironment::aggregated_fitness(particle_cnt as u16);
                    // let g_fitness = 1; // added
                    g_fitness as f64 / (genome_env.max_edge_cnt as f64)
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

    fn run_through(&mut self) {
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
    //size = [(10,7),(13,9),(20,14),(27,19)]
    let mut ga_sops = GeneticAlgo::init_ga(50, 100, 0, 0.16, 20, true, [(6,4),(10,7),(13,9)].to_vec(), [31728, 26812, 73921, 92031, 84621].to_vec());
    ga_sops.run_through();

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
    // let genome = [
    //     8,
    //     4,
    //     2,
    //     1,
    //     1,
    //     1
    // ];
    // println!("{:?}", genome);

    // let mut sops_trial = SOPSEnvironment::init_sops_env(&genome, 13, 9, 31728);
    // sops_trial.print_grid();
    // let edge_cnt: u64 = sops_trial.evaluate_fitness().into();
    // println!("Edge Count: {}", edge_cnt);
    // println!("Max Edge Count: {}", sops_trial.max_edge_cnt);
    // println!("Starting Fitness: {}", edge_cnt as f32/ sops_trial.max_edge_cnt as f32);
    // println!("No. of Participants {}", sops_trial.participants.len());
    // let now = Instant::now();
    // let edge_cnt: u64 = sops_trial.simulate().into();
    // let elapsed = now.elapsed().as_secs();
    // sops_trial.print_grid();
    // println!("Edge Count: {}", edge_cnt);
    // println!("Fitness: {}", edge_cnt as f32/ sops_trial.max_edge_cnt as f32);
    // println!("Trial Elapsed Time: {:.2?}", elapsed);
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
