use super::Particle;

use rand::SeedableRng;
use rand::{distributions::Bernoulli, distributions::Open01, distributions::Uniform, rngs, Rng};
use nanorand::{Rng as NanoRng, WyRand};
use rand_distr::num_traits::Pow;
use std::usize;

pub struct SOPSegEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [[[u16; 6]; 7];7],
    sim_duration: u64,
    fitness_val: f64,
    size: usize,
    max_fitness: u64,
    arena_layers: u16,
    particle_layers: u16,
    phenotype_sum: u16
}


impl SOPSegEnvironment {
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

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    // Use a Random Seed value to get the same random init config
    pub fn init_sops_env(genome: &[[[u16; 6]; 7]; 7], arena_layers: u16, particle_layers: u16, seed: u64) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let num_particles = 6*particle_layers*(1+particle_layers)/2;
        let num_particles_clr = num_particles/3;
        let agg_edge_cnt: u64 = (3*(3*particle_layers.pow(2)+particle_layers-1)).into();
        let agg_clr_edge_cnt: u64 = (3*particle_layers.pow(2)-particle_layers-1).into();
        let mut grid_rng = SOPSegEnvironment::seed_rng(seed);
        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = 4;
                grid[(i+arena_layers+j) as usize][i as usize] = 4;
                j +=1;
            }
        }
        let mut current_color_cnt = 0;
        let mut current_color = 1;
        //init grid and particles
        while participants.len() < num_particles.into() {
            let i = grid_rng.sample(&SOPSegEnvironment::grid_rng(grid_size));
            let j = grid_rng.sample(&SOPSegEnvironment::grid_rng(grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    color: current_color
                });
                grid[i][j] = current_color;
                current_color_cnt +=1;
                if current_color_cnt == num_particles_clr {
                    current_color_cnt = 0;
                    current_color +=1;
                }
            }
            
        }

        SOPSegEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_fitness: agg_edge_cnt + 3*(agg_clr_edge_cnt),
            arena_layers,
            particle_layers,
            phenotype_sum: genome.iter().map(|y| -> u16 { y.iter().map(|x| -> u16 { x.iter().sum() }).sum() }).sum()
        }
    }

    pub fn print_grid(&self) {
        println!("SOPS grid");
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                print!(" {} ", self.grid[i][j]);
            }
            println!("");
        }
    }

    fn get_neighbors_cnt(&self, i: u8, j: u8, n_clr: u8) -> (u8, u8, u8) {
        let mut cnt = 0;
        let mut same_clr_cnt = 0;
        let mut n_clr_cnt = 0;
        for idx in 0..6 {
            let new_i = (i as i32 + SOPSegEnvironment::directions()[idx].0) as usize;
            let new_j = (j as i32 + SOPSegEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                if self.grid[new_i][new_j] != 0 && self.grid[new_i][new_j] != 4 {
                    cnt += 1;
                    if self.grid[new_i][new_j] == self.grid[i as usize][j as usize] {
                        same_clr_cnt += 1;
                    } else if self.grid[new_i][new_j] == n_clr {
                        n_clr_cnt += 1;
                    }
                }
            }
        }
        // NOT_NEEDED: reduce neighbor color count by 1 since we don't want to consider the neighbor itself
        (cnt, same_clr_cnt, n_clr_cnt)
    }

    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // not need to move/swap if the particle is swapping with a particle of same color or is going out of bounds
            if self.grid[new_i][new_j] == 4 || self.grid[new_i][new_j] == self.grid[particle.x as usize][particle.y as usize] {
                return false;
            } else if self.grid[new_i][new_j] == 0 {
                // simple move in empty location
                // println!("Particle at {},{},c{} moves to {},{}", particle.x, particle.y, particle.color, new_i, new_j);
                self.grid[particle.x as usize][particle.y as usize] = 0;
                self.grid[new_i][new_j] = particle.color;
                let mut particle_mut = &mut self.participants[particle_idx];
                particle_mut.x = new_i as u8;
                particle_mut.y = new_j as u8;
                return true;
            }
            else {
                // swap with another particle
                // println!("Particle at {},{},c{} swaps with {},{},c{}", particle.x, particle.y, particle.color, new_i, new_j, self.grid[new_i][new_j]);
                let swap_idx = self.participants.iter().position(|par| par.x as usize == new_i && par.y as usize == new_j).unwrap();
                self.grid[particle.x as usize][particle.y as usize] = self.grid[new_i][new_j];
                self.grid[new_i][new_j] = particle.color;
                let temp_x = particle.x;
                let temp_y = particle.y;
                let mut particle_mut = &mut self.participants[particle_idx];
                particle_mut.x = new_i as u8;
                particle_mut.y = new_j as u8;
                let mut particle_swap = &mut self.participants[swap_idx];
                particle_swap.x = temp_x;
                particle_swap.y = temp_y;
                return true;
            }
        } else {
            return false;
        }
    }

    fn move_particles(&mut self, cnt: usize) {
        let mut par_moves: Vec<(usize, (i32, i32))> = Vec::new();

         for _ in 0..cnt {
            let par_idx = SOPSegEnvironment::rng().sample(&self.unfrm_par());
            let particle: &Particle = &self.participants[par_idx];
            let move_dir = SOPSegEnvironment::directions()
                        [SOPSegEnvironment::rng().sample(&SOPSegEnvironment::unfrm_dir())];
            let new_i = (particle.x as i32 + move_dir.0) as usize;
            let new_j = (particle.y as i32 + move_dir.1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            if self.grid[new_i][new_j] == 4 || self.grid[new_i][new_j] == self.grid[particle.x as usize][particle.y as usize] {
                continue;
            }
            let (egde_cnt, clr_edge_cnt, n_clr_cnt) = self.get_neighbors_cnt(particle.x, particle.y, self.grid[new_i][new_j]);
            if clr_edge_cnt < 6 {
                let move_prb: f64 =
                self.phenotype[egde_cnt as usize][n_clr_cnt as usize][clr_edge_cnt as usize] as f64 / (self.phenotype_sum as f64);
                // if SOPSegEnvironment::move_nrng().generate_range(1_u64..=1000)
                //     <= (move_prb * 1000.0) as u64
                // {
                //     let move_dir = SOPSegEnvironment::directions()
                //         [SOPSegEnvironment::move_nrng().generate_range(1..6)];
                //     par_moves.push((par_idx, move_dir));
                // }
                if SOPSegEnvironment::move_frng().u64(1_u64..=10000)
                    <= (move_prb * 10000.0) as u64
                {
                    par_moves.push((par_idx, move_dir));
                }
            }
        }
        }

        // Parallel execution
        /*
        let par_moves: Vec<(usize, (i32, i32))> = (0..cnt).into_par_iter().filter_map(|_| {
            let par_idx = SOPSegEnvironment::move_frng().usize(0..self.participants.len());
            let particle: &Particle = &self.participants[par_idx];
            let n_cnt = self.get_neighbors_cnt(particle.x, particle.y) as usize;
            if n_cnt == 6 {
                return None;
            }
            let move_prb: f64 =
                self.phenotype[n_cnt] as f64 / (self.phenotype.iter().sum::<u16>() as f64);
            // if SOPSegEnvironment::move_nrng().generate_range(1_u64..=1000)
            //     <= (move_prb * 1000.0) as u64
            // {
            //     let move_dir = SOPSegEnvironment::directions()
            //         [SOPSegEnvironment::move_nrng().generate_range(1..6)];
            //     par_moves.push((par_idx, move_dir));
            // }
            if SOPSegEnvironment::move_frng().u64(1_u64..=1000)
                <= (move_prb * 1000.0) as u64
            {
                let move_dir = SOPSegEnvironment::directions()
                    [SOPSegEnvironment::move_frng().usize(1..6)];
                return Some((par_idx, move_dir));
            }
            return None;
        }).collect();
         */

        for moves in par_moves.iter() {
            self.move_particle_to(moves.0, moves.1);
        }
    }

    pub fn evaluate_fitness(&self) -> f32 {
        let mut clr_edges = [0_u32; 3];
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            let neigbor_edges = self.get_neighbors_cnt(particle.x, particle.y, 0);
            clr_edges[(particle.color-1) as usize] += neigbor_edges.1 as u32;
            sum + neigbor_edges.0 as u32
        });
        ((edges as f32) + (clr_edges.iter().sum::<u32>() as f32)) / 2.0
    }

    pub fn simulate(&mut self, take_snaps: bool) -> f32 {
        for step in 0..self.sim_duration {
            // let now = Instant::now();
            self.move_particles(1 as usize);
            // let elapsed = now.elapsed().as_micros();
            // println!("Step Elapsed Time: {:.2?}", elapsed);
            if take_snaps && (step == (self.participants.len() as u64) || step == (self.participants.len() as u64).pow(2) || step == (self.participants.len() as u64).pow(3) || step == (self.participants.len() as u64).pow(3)*2|| step == (self.participants.len() as u64).pow(3)*3|| step == (self.participants.len() as u64).pow(3)*4|| step == (self.participants.len() as u64).pow(3)*5) {
                self.print_grid();
                println!("Step {}", step);
                // println!("No. of Participants {:?}", self.get_participant_cnt());
                // let particles_cnt = self.get_participant_cnt();
                // if particles_cnt.iter().any(|&x| x != (self.participants.len() as u16/3)) {
                //     panic!("Something is wrong");
                // }
                println!("Edge Count: {}", self.evaluate_fitness());
                println!("Fitness: {}", self.evaluate_fitness() as f32/ self.get_max_fitness() as f32);
            }
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness as f64;
        fitness
    }

    pub fn get_max_fitness(&self) -> u64 {
        self.max_fitness
    }

    pub fn get_participant_cnt(&self) -> [u16; 3] {
        let mut clr_particles = [0_u16; 3];
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                if self.grid[i][j] != 0 && self.grid[i][j] != 4 {
                    clr_particles[(self.grid[i][j]-1) as usize] += 1
                }
            }
        }
        // self.participants.iter().for_each(|p| clr_particles[(p.color-1) as usize] +=1);
        clr_particles
    }
}
