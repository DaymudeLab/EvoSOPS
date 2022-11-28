use rand::{distributions::Uniform, rngs, Rng};
use rand_distr::Normal;
use rayon::prelude::*;
use std::convert::TryInto;

struct Particle {
    x: u8,
    y: u8,
}
#[derive(Debug, Copy, Clone)]
struct Genome {
    string: [f64; 6],
    fitness: f64,
}

struct SOPSEnvironment {
    grid: [[bool; 18]; 18],
    participants: Vec<Particle>,
    phenotype: [f64; 6],
    sim_duration: u64,
    fitness_val: f64,
}

struct GeneticAlgo {
    max_gen: u16,
    trial_cnt: u16,
    elitist_cnt: u16,
    population: Vec<Genome>,
    mut_rate: f64,
    mut_mu: f64,
    mut_sd: f64,
}

impl SOPSEnvironment {
    const SIZE: usize = 18;
    const par_density: f64 = 0.5;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn grid_rng() -> Uniform<usize> {
        Uniform::new(0, SOPSEnvironment::SIZE)
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

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    fn static_init(genome: &[f64; 6]) -> Self {
        let grid: [[bool; 18]; 18] = [
            [
                true, true, false, true, false, false, false, true, false, false, false, true,
                true, false, false, false, false, false,
            ],
            [
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, true, false, false, false, true,
            ],
            [
                false, false, true, false, false, true, false, true, false, true, true, true,
                false, false, true, false, false, false,
            ],
            [
                false, true, true, true, false, false, true, true, true, false, false, false,
                false, false, false, true, true, true,
            ],
            [
                true, false, true, true, false, true, false, true, true, true, false, false, false,
                true, false, false, true, false,
            ],
            [
                true, false, true, false, false, false, false, true, false, false, false, false,
                true, true, true, true, true, false,
            ],
            [
                false, false, false, true, true, false, true, false, false, true, true, true,
                false, false, false, false, true, false,
            ],
            [
                false, false, false, false, true, true, true, false, true, false, false, false,
                true, false, true, false, false, false,
            ],
            [
                false, false, false, true, false, false, false, false, false, true, false, false,
                false, false, false, false, true, true,
            ],
            [
                true, true, false, true, false, false, true, true, false, true, false, true, false,
                false, false, false, false, true,
            ],
            [
                false, true, true, false, false, false, true, false, false, true, false, true,
                true, true, false, false, false, false,
            ],
            [
                true, false, false, true, true, false, false, false, false, false, true, true,
                false, false, false, false, true, true,
            ],
            [
                false, true, false, false, false, true, false, false, true, false, true, true,
                true, true, false, true, true, false,
            ],
            [
                true, true, false, false, false, true, false, false, true, true, false, false,
                true, false, false, true, true, true,
            ],
            [
                true, true, true, true, true, false, true, true, true, false, true, false, true,
                false, true, false, false, false,
            ],
            [
                false, false, true, false, true, true, false, true, false, false, false, false,
                true, false, true, false, false, false,
            ],
            [
                true, false, false, false, false, true, true, true, true, true, true, false, true,
                true, false, false, true, false,
            ],
            [
                false, true, true, false, false, false, false, true, true, false, false, true,
                false, true, false, false, false, false,
            ],
        ];
        let mut participants: Vec<Particle> = vec![];
        //init grid and particles
        for i in 0..SOPSEnvironment::SIZE {
            for j in 0..SOPSEnvironment::SIZE {
                if grid[i][j] {
                    participants.push(Particle {
                        x: i as u8,
                        y: j as u8,
                    });
                }
            }
        }

        let particle_cnt = participants.len();

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (particle_cnt as u64).pow(3),
            fitness_val: 0.0,
        }
    }

    fn init_sops_env(genome: &[f64; 6]) -> Self {
        let mut grid: [[bool; 18]; 18] = [[false; 18]; 18];
        let mut participants: Vec<Particle> = vec![];
        let grid_size = SOPSEnvironment::SIZE * SOPSEnvironment::SIZE;
        let num_particles = ((grid_size as f64) * SOPSEnvironment::par_density) as u64;
        //init grid and particles
        for _ in 0..num_particles {
            let i = SOPSEnvironment::rng().sample(&SOPSEnvironment::grid_rng());
            let j = SOPSEnvironment::rng().sample(&SOPSEnvironment::grid_rng());
            participants.push(Particle {
                x: i as u8,
                y: j as u8,
            });
            grid[i][j] = true;
        }

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: num_particles.pow(3),
            fitness_val: 0.0,
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
                if self.grid[new_i][new_j] {
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
            if self.grid[new_i][new_j] {
                return false;
            } else {
                self.grid[particle.x as usize][particle.y as usize] = false;
                self.grid[new_i][new_j] = true;
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
            let move_prb = self.phenotype[n_cnt];
            if SOPSEnvironment::rng().sample(&SOPSEnvironment::unfrm_move())
                <= (move_prb * 1000.0) as u64
            {
                let move_dir = SOPSEnvironment::directions()
                    [SOPSEnvironment::rng().sample(&SOPSEnvironment::unfrm_dir())];
                par_moves.push((par_idx, move_dir));
            }
        }

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
            self.move_particles(4);
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness as f64;
        fitness
    }
}

impl GeneticAlgo {
    #[inline]
    fn genome_init_rng() -> Normal<f64> {
        Normal::new(0.5, 0.1).unwrap()
    }

    #[inline]
    fn unfrm_100() -> Uniform<u16> {
        Uniform::new_inclusive(1, 100)
    }

    fn mut_val(&self) -> Normal<f64> {
        Normal::new(self.mut_mu, self.mut_sd).unwrap()
    }

    #[inline]
    pub fn init_ga(
        population_size: u16,
        max_gen: u16,
        trial_cnt: u16,
        elitist_cnt: u16,
        mut_rate: f64,
        mut_mu: f64,
        mut_sd: f64,
    ) -> Self {
        let mut starting_pop: Vec<Genome> = vec![];

        for _ in 0..population_size {
            //init genome
            let genome: [f64; 6] = TryInto::try_into(
                (0..6)
                    .map(|_| SOPSEnvironment::rng().sample(GeneticAlgo::genome_init_rng()))
                    .collect::<Vec<f64>>(),
            )
            .unwrap();
            starting_pop.push(Genome {
                string: (genome),
                fitness: (0.0),
            });
        }

        GeneticAlgo {
            max_gen,
            trial_cnt,
            elitist_cnt,
            population: starting_pop,
            mut_rate,
            mut_mu,
            mut_sd,
        }
    }

    fn mutate_genome(&self, genome: &[f64; 6]) -> [f64; 6] {
        let mut new_genome = genome.clone();
        //mutate genome
        for i in 0..genome.len() {
            let smpl = SOPSEnvironment::rng().sample(&GeneticAlgo::unfrm_100());
            if smpl as f64 <= self.mut_rate * 100.0 {
                let perturb = SOPSEnvironment::rng().sample(self.mut_val());
                new_genome[i] = (genome[i] + perturb).clamp(0.0, 1.0);
            }
        }
        new_genome
    }

    fn generate_new_pop(&mut self) {
        let mut new_pop: Vec<Genome> = vec![];
        let mut selected_g: Vec<[f64; 6]> = vec![];
        let mut rank_wheel: Vec<usize> = vec![];
        self.population.sort_unstable_by(|genome_a, genome_b| {
            genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
        });

        //selection
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
        //perform selection
        let rank_wheel_rng = Uniform::new(0, rank_wheel.len());
        for _ in 0..(self.population.len() - self.elitist_cnt as usize) {
            let wheel_idx = SOPSEnvironment::rng().sample(&rank_wheel_rng);
            let genome_idx = rank_wheel[wheel_idx];
            selected_g.push(self.population[genome_idx].string);
        }
        //perform mutation
        for idx in 0..selected_g.len() {
            let genome = selected_g[idx];
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
        let trials = self.trial_cnt;
        self.population.par_iter_mut().for_each(|genome| {
            //bypass if genome has already fitness value calculated
            if gen > 0 && genome.fitness > 0.0 {
                return;
            }
            let genome_s = genome.string.clone();
            let mut fitness_t = 0.0;
            for _ in 0..trials {
                let mut genome_env = SOPSEnvironment::static_init(&genome_s);
                let g_fitness = genome_env.simulate();
                fitness_t += g_fitness as f64;
            }

            genome.fitness = fitness_t / (trials as f64) as f64;
        });
        //avg.fitness of old population
        let fit_sum = self
            .population
            .iter()
            .fold(0.0, |sum, genome| sum + genome.fitness);
        println!(
            "Avg. Fitness -> {}",
            fit_sum / (self.population.len() as f64)
        );
        let mut pop_dist: Vec<f64> = vec![];
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = self.population[i];
                let genome2 = self.population[j];
                let mut dis_sum = 0.0;
                for idx in 0..genome1.string.len() {
                    let dis = genome1.string[idx] - genome2.string[idx];
                    dis_sum += dis.powf(2.0);
                }
                pop_dist.push(dis_sum.sqrt());
            }
        }
        let pop_diversity: f64 = pop_dist.iter().sum();
        println!(
            "Population diversity -> {}",
            pop_diversity / (pop_dist.len() as f64)
        );
        //generate new population
        self.generate_new_pop();
        let best_genome = self.population[0];
        println!("Best Genome -> {best_genome:?}");
    }

    fn run_through(&mut self) {
        for gen in 0..self.max_gen {
            println!("Starting Gen:{}", gen);
            self.step_through(gen);
        }
        let best_genome = self.population[0];
        let mut best_genome_env = SOPSEnvironment::static_init(&best_genome.string);
        best_genome_env.print_grid();
        let g_fitness = best_genome_env.simulate();
        best_genome_env.print_grid();
        println!("Best genome's fitness is {}", g_fitness);
        println!("{best_genome:?}");
    }
}

fn main() {
    let mut ga_sops = GeneticAlgo::init_ga(20, 50, 3, 3, 0.07, 0.0, 0.05);
    ga_sops.run_through();
    // let genome = [
    //     0.879410923312898,
    //     0.7072823455008604,
    //     0.3758316160483933,
    //     0.4947528743281018,
    //     0.32321085020900764,
    //     0.6018268442926183,
    // ];
    // let mut sops_trial = SOPSEnvironment::init_sops_env(&genome);
    // sops_trial.print_grid();
    // println!("{}", sops_trial.evaluate_fitness());
    // println!("No. of Participants {}", sops_trial.participants.len());
    // let mut sops_trial_2 = SOPSEnvironment::static_init(&genome);
    // sops_trial_2.print_grid();
    // println!("{}", sops_trial_2.evaluate_fitness());
    // println!("{}",sops_trial.evaluate_fitness());
    // println!("{}",sops_trial.simulate());
    // sops_trial.print_grid();

    // let mut grid: [[bool; 18]; 18] = [[false; 18]; 18];
    // let init_nrm = Normal::new(0.5, 0.1).unwrap();
    // let mut rng = rand::thread_rng();
    // let grid_rng = Uniform::new(0, grid.len());

    // println!("A simple Grid\n");

    // println!("{:?}", genome);

    // println!("{:?}", genome);

    // let v = rng.sample(&init_nrm);
    // println!("{} is from a N(0.5, 0.1) distribution", v);
    // println!("{} particles in grid of {} vertices",(((grid.len()*grid.len()) as f32)*0.3) as u64, grid.len()*grid.len());
}
