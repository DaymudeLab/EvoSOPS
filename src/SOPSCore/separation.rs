use super::Particle;

use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};
use std::usize;
use std::collections::HashMap;

/*
 * Main Class for the Separation Behaviour Experiments on SOPS grid
 * Defines how the genome is interpreted and how each transition of
 * particles is derived from the the genome. Also provides final SOPS
 * grid evaluations to assess the fitness score of the genome
 *  */
pub struct SOPSepEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [[[u8; 10]; 6]; 10],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    max_fitness_c1: f32,
    max_fitness_c2: f32,
    arena_layers: u16,
    particle_layers: u16,
    granularity: u8,
    lookup_dim_idx: HashMap<(u8, u8, u8), u8>,
    w1: f32,
    w2: f32
}


impl SOPSepEnvironment {
    const EMPTY: u8 = 0;
    const BOUNDARY: u8 = 4;

    const BACK: u8 = 0;
    const MID: u8 = 1;
    const FRONT: u8 = 2;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn seed_rng(seed: u64) -> rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    #[inline]
    fn move_frng() -> fastrand::Rng {
        fastrand::Rng::new()
    }

    #[inline]
    fn grid_rng(size_s: usize, size_e: usize) -> Uniform<usize> {
        Uniform::new(size_s, size_e)
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

    // granularity is 10
    #[inline]
    fn gene_probability() -> Vec<u16> {
        vec![1000, 500, 250, 125, 63, 31, 16, 8, 4, 2, 1]
    }
    
    // granularity is 15
    #[inline]
    fn theory_gene_probability() -> Vec<u16> {
        vec![1000, 667, 444, 296, 250, 167, 111, 74, 63, 42, 28, 19, 16, 10, 7, 5]
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    /*
     * Initialize a SOPS grid and place particles based on particle layer and arena layer count
     * Parameters Particle layers and Arena layers refer to the complete hexagonal lattice layers
     * of the SOPS grid and this also defines the total density of particles in the arena.
     * Calculates Max edge count possible for all the particles irrespective of the color
     * Calculates Max edge count possible for all the particles of the same color
     * Also accept the weights for Agg and Sep components
     * NOTE: Use the Same random Seed value to get the same random init config
     *  */
    pub fn init_sops_env(genome: &[[[u8; 10]; 6]; 10], arena_layers: u16, particle_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        // let init_compartment_layers = arena_layers-10;
        // let init_compartment_size = ((particle_layers)*2 + 1) as usize;
        // let init_compartment_start = (grid_size/4) as usize;
        // let init_compartment_end = init_compartment_start + init_compartment_size;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let num_particles = 6*particle_layers*(1+particle_layers)/2;
        let num_particles_clr = num_particles/3;
        
        // No. of edges in the aggregated config with one less particle than aggregation behavior
        let agg_edge_cnt: f32 = (3*(3*particle_layers.pow(2)+particle_layers-1)).into();
        // No. of edges in the aggregated config for each color
        let agg_clr_edge_cnt: f32 = (3*particle_layers.pow(2)-particle_layers-1).into();
        
        let mut grid_rng = SOPSepEnvironment::seed_rng(seed);
        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = SOPSepEnvironment::BOUNDARY;
                grid[(i+arena_layers+j) as usize][i as usize] = SOPSepEnvironment::BOUNDARY;
                j +=1;
            }
        }
        let mut current_color_cnt = 0;
        let mut current_color = 1;
        //init grid and particles
        while participants.len() < num_particles.into() {
            // let i = grid_rng.sample(&SOPSepEnvironment::grid_rng(init_compartment_start,init_compartment_end));
            // let j = grid_rng.sample(&SOPSepEnvironment::grid_rng(init_compartment_start,init_compartment_end));
            let i = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,grid_size));
            let j = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    state: current_color
                });
                grid[i][j] = current_color;
                current_color_cnt +=1;
                if current_color_cnt == num_particles_clr {
                    current_color_cnt = 0;
                    current_color +=1;
                }
            }   
        }

        // let mut current_color = 0;

        // // Theory
        // while participants.len() < num_particles.into() {
        //     // println!("Placing par:{}",participants.len());
        //     if participants.len() == 0 {
        //         let i = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,grid_size));
        //         let j = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,grid_size));
        //         if grid[i][j] == 0 {
        //             current_color = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,3)) as u8;
        //             participants.push(Particle {
        //                 x: i as u8,
        //                 y: j as u8,
        //                 state: current_color + 1
        //             });
        //             grid[i][j] = current_color + 1;
        //         }
        //     }
        //     else {
        //         // check if prev par has any space
        //         let par_idx = SOPSepEnvironment::move_frng().usize(..participants.len());
        //         let particle = &participants[par_idx];
        //         let mut place = vec![];
        //         for idx in 0..6 {
        //             let new_i = (particle.x as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
        //             let new_j = (particle.y as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
        //             if (0..grid.len()).contains(&new_i) && (0..grid.len()).contains(&new_j) {
        //                 if grid[new_i][new_j] == 0 {
        //                     place.push((new_i, new_j))
        //                 }
        //             }
        //         }
        //         // 
        //         if place.len() > 2 {
        //             let move_dir = place
        //                 [SOPSepEnvironment::move_frng().usize(..place.len())];
        //             let i = move_dir.0;
        //             let j = move_dir.1;
        //             if (0..(grid.len())).contains(&i) && (0..(grid.len())).contains(&j) {
        //                 if grid[i][j] == 0 {
        //                     current_color = grid_rng.sample(&SOPSepEnvironment::grid_rng(0,3)) as u8;
        //                     participants.push(Particle {
        //                         x: i as u8,
        //                         y: j as u8,
        //                         state: current_color + 1
        //                     });
        //                     grid[i][j] = current_color + 1;
        //                 }
        //             }
        //         }
        //     }               
        // }

        // TODO: Make this a static const variable
        // Mapping table for various intra group(F/M/B) configurations -> index in genome's dimension
        // intra group(F/M/B) configurations ie. all_cnt, same_clr_cnt, all_possible_cnt(static position cnt in F/M/B — (3/2/3))
        let lookup_dim_idx: HashMap<(u8, u8, u8), u8> = ([
            ((0,0,2), 0),
            ((1,1,2), 1),
            ((1,0,2), 2),
            ((2,2,2), 3),
            ((2,1,2), 4),
            ((2,0,2), 5),
            ((0,0,3), 0),
            ((1,1,3), 1),
            ((1,0,3), 2),
            ((2,2,3), 3),
            ((2,1,3), 4),
            ((2,0,3), 5),
            ((3,3,3), 6),
            ((3,2,3), 7),
            ((3,1,3), 8),
            ((3,0,3), 9),
        ]).into();

        SOPSepEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_fitness_c1: agg_edge_cnt,
            max_fitness_c2: agg_clr_edge_cnt,
            arena_layers,
            particle_layers,
            granularity,
            lookup_dim_idx,
            w1,
            w2
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

    /*
     * Func to calculate a particle's neighbor count both for (same and any color)
     *  */
    fn get_neighbors_cnt(&self, i: u8, j: u8, n_clr: u8) -> (u8, u8, u8) {
        let mut cnt = 0;
        let mut same_clr_cnt = 0;
        let mut n_clr_cnt = 0;
        for idx in 0..6 {
            let new_i = (i as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
            let new_j = (j as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
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

    /*
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // TODO: remove these checks since move is already check for possibility
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // not need to move/swap if the particle is swapping with a particle of same color or is going out of bounds
            if self.grid[new_i][new_j] == SOPSepEnvironment::BOUNDARY || self.grid[new_i][new_j] == self.grid[particle.x as usize][particle.y as usize] {
                return false;
            } else if self.grid[new_i][new_j] == SOPSepEnvironment::EMPTY {
                // simple move in empty location
                // println!("Particle at {},{},c{} moves to {},{}", particle.x, particle.y, particle.color, new_i, new_j);
                self.grid[particle.x as usize][particle.y as usize] = SOPSepEnvironment::EMPTY;
                self.grid[new_i][new_j] = particle.state;
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
                self.grid[new_i][new_j] = particle.state;
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

    /*
     * Func to get index into a genome's dimension
     */
    fn get_dim_idx(&self, all_cnt: u8, same_cnt: u8, all_possible_cnt: u8) -> u8 {

        match self.lookup_dim_idx.get(&(all_cnt, same_cnt, all_possible_cnt)) {
            Some(idx) => {
                return *idx;
            }
            None => {0},
        }
    }

    /*
     * Func to calculate a particle's connectivity (used in theory-based simulation)
     *  */
     fn check_connectivity(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let mut back_cnt: u8 = 0;
        let mut mid_cnt: u8 = 0;
        let mut front_cnt: u8 = 0;
        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        let mut back_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        let mut front_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        let mut mid_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == move_i) & (new_j == move_j)) {
                // print!("{}",idx);
                seen_neighbor_cache.insert([new_i, new_j], true);
                // all_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY  {
                    back_cnt += 1;
                    back_neighbor_cache.insert([new_i, new_j], true);
                    if particle.state == self.grid[new_i][new_j] {
                        // back_same_clr_cnt += 1;
                    }
                }
            }
        }
        // print!("\t");
        // Neighborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                // print!("{}",idx);
                let mut position_type = SOPSepEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSepEnvironment::MID;
                    }
                    None => {
                    },
                }
                if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY {
                    match position_type {
                        SOPSepEnvironment::FRONT => {
                            front_neighbor_cache.insert([new_i, new_j], true);
                            front_cnt += 1;
                            if particle.state == self.grid[new_i][new_j] {
                                // front_same_clr_cnt += 1;
                            }
                        }
                        SOPSepEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                            mid_neighbor_cache.insert([new_i, new_j], true);
                            back_neighbor_cache.remove(&[new_i, new_j]);
                            if particle.state == self.grid[new_i][new_j] {
                                // mid_same_clr_cnt += 1;
                                // back_same_clr_cnt -= 1;
                            }
                        }
                        _ => todo!()
                    }
                }
            }
        }

        if mid_neighbor_cache.keys().len() as u8 != mid_cnt {
            println!("Error Mid")   
        }

        if back_neighbor_cache.keys().len() as u8 != back_cnt {
            println!("Error back")   
        }

        if front_neighbor_cache.keys().len() as u8 != front_cnt {
            println!("Error front")   
        }
        
        if mid_cnt > 0 {
            // five neighbor check
            if (mid_cnt == 2) && (back_cnt == 3) {
                return false;
            }

            // /*
            // check if back and front are connected to mid
                // seen_neighbor_cache == back cache
                let mut mid_touch = false;
                for loc in back_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) && (0..self.grid.len()).contains(&new_j) && !((new_i == particle.x.into()) && (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match back_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                            match mid_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    mid_touch = true;
                                    neighbor_touch = true
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if neighbor_touch == false {
                        return false;
                    }
                }
                // if none of back touch mid
                if mid_touch == false {
                    return false;
                }

                // same for front
                let mut mid_touch_front = false;
                for loc in front_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) && (0..self.grid.len()).contains(&new_j) && !((new_i == particle.x.into()) && (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match front_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                            match mid_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    mid_touch_front = true;
                                    neighbor_touch = true
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if neighbor_touch == false {
                        return false;
                    }
                }
                // if none of back touch mid
                if mid_touch_front == false {
                    return false;
                }
            //  */
            /*
            // Property 1:
            if back_cnt == 1 {
                let flip_direction = (direction.0 * -1, direction.1 * -1);
                let new_i = (particle.x as i32 + flip_direction.0) as usize;
                let new_j = (particle.y as i32 + flip_direction.1) as usize;
                if (0..self.grid.len()).contains(&new_i) && (0..self.grid.len()).contains(&new_j) {
                    if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY  {
                        return false;
                    }
                }
            }

            if front_cnt == 1 {
                let new_i = (move_i as i32 + direction.0) as usize;
                let new_j = (move_j as i32 + direction.1) as usize;
                if (0..self.grid.len()).contains(&new_i) && (0..self.grid.len()).contains(&new_j) {
                    if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY  {
                        return false;
                    }
                }
            }

            if mid_cnt == 1 {
                // check if back and front are connected to mid
                let mut mid_touch = false;
                for loc in back_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match back_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                            match mid_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    mid_touch = true;
                                    neighbor_touch = true
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if !neighbor_touch {
                        return false;
                    }
                }
                // if none of back touch mid
                if !mid_touch {
                    return false;
                }

                // same for front
                let mut mid_touch_front = false;
                for loc in front_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match front_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                            match mid_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    mid_touch_front = true;
                                    neighbor_touch = true
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if !neighbor_touch {
                        return false;
                    }
                }
                // if none of back touch mid
                if !mid_touch_front {
                    return false;
                }
            }
            */
            return true;
        }
        else if (mid_cnt == 0) && (back_cnt > 0) && (front_cnt > 0) {
            // Property 2
            if back_cnt == 2 {
                for loc in back_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match back_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if neighbor_touch == false {
                        return false;
                    }
                }
            }
            if front_cnt == 2 {
                for loc in front_neighbor_cache.keys() {
                    // for every particle in back
                    let mut neighbor_touch = false;
                    for idx in 0..6 {
                        let new_i = (loc[0] as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
                        let new_j = (loc[1] as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
                        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                            // print!("{}",idx);
                            match front_neighbor_cache.get(&[new_i, new_j]) {
                                Some(_exists) => {
                                    neighbor_touch = true;
                                }
                                None => {},
                            }
                        }
                    }
                    // if a particle has no neighbors
                    if neighbor_touch == false {
                        return false;
                    }
                }
            }
            return true;
        }
        else {
            return false;
        }
    }

    /*
     * Func to calculate a particle's extended neighbor count
     *  */
     fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8) {
        let mut back_cnt: u8 = 0;
        let mut back_same_clr_cnt: u8 = 0;
        let mut mid_cnt: u8 = 0;
        let mut mid_same_clr_cnt: u8 = 0;
        let mut front_cnt: u8 = 0;
        let mut front_same_clr_cnt: u8 = 0;
        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == move_i) & (new_j == move_j)) {
                // print!("{}",idx);
                seen_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY  {
                    back_cnt += 1;
                    if particle.state == self.grid[new_i][new_j] {
                        back_same_clr_cnt += 1;
                    }
                }
            }
        }
        // print!("\t");
        // Neighborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSepEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSepEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                // print!("{}",idx);
                let mut position_type = SOPSepEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSepEnvironment::MID;
                    }
                    None => {},
                }
                if self.grid[new_i][new_j] != SOPSepEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSepEnvironment::BOUNDARY {
                    match position_type {
                        SOPSepEnvironment::FRONT => {
                            front_cnt += 1;
                            if particle.state == self.grid[new_i][new_j] {
                                front_same_clr_cnt += 1;
                            }
                        }
                        SOPSepEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                            if particle.state == self.grid[new_i][new_j] {
                                mid_same_clr_cnt += 1;
                                back_same_clr_cnt -= 1;
                            }
                        }
                        _ => todo!()
                    }
                }
            }
        }
        // print!("\t");
        let back_idx: u8 = self.get_dim_idx(back_cnt, back_same_clr_cnt, 3);
        let mid_idx: u8 = self.get_dim_idx(mid_cnt, mid_same_clr_cnt, 2);
        let front_idx: u8 = self.get_dim_idx(front_cnt, front_same_clr_cnt, 3);
        (back_idx.clamp(0, 9), mid_idx.clamp(0, 5), front_idx.clamp(0, 9))
    }

    /*
     * Func to check if the proposed move is possible or not for a particle
     *  */
     fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> u8 {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // Move particle if movement is within grid array's bound
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // check to see if move is valid ie. within arena bounds
            if self.grid[new_i][new_j] == SOPSepEnvironment::BOUNDARY {
                return 0;
            } else {
                // can move the particle
                if self.grid[new_i][new_j] == SOPSepEnvironment::EMPTY {
                    return 2;
                }
                // can swap with the neighbor
                return 1;
            }
        } else {
            return 0;
        }
    }

    /*
     * Func to get adjacent particle's index in participants vector
     *  */
     fn get_adjacent_particle(&self, particle_idx: usize, direction: (i32, i32)) -> usize {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as u8;
        let new_j = (particle.y as i32 + direction.1) as u8;
        let adjacent_particle_idx = &self.participants.iter().position(|par| par.x == new_i && par.y == new_j).unwrap();
        return *adjacent_particle_idx;
    }

    /*
     * Func to move 'n' particles in random directions in the SOPS grid
     *  */
    fn move_particles(&mut self, cnt: usize) {
        
        // Choose a random particle for movement
        let par_idx = SOPSepEnvironment::move_frng().usize(..self.participants.len());
        // Choose a direction at random (out of the 6)
        let move_dir = SOPSepEnvironment::directions()
                        [SOPSepEnvironment::move_frng().usize(..SOPSepEnvironment::directions().len())];
        
        match self.particle_move_possible(par_idx, move_dir) {
            1 => {
                // swap
                // Get the neighborhood configuration
                let (back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);
                // Move basis probability given by the genome for moving for given configuration
                let move_prb_p1 = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize];
    
                let par_2_idx = &self.get_adjacent_particle(par_idx, move_dir);
                let flip_direction = (move_dir.0 * -1, move_dir.1 * -1);
                let (back_cnt_2, mid_cnt_2, front_cnt_2) = self.get_ext_neighbors_cnt(*par_2_idx, flip_direction);
                let move_prb_p2 = self.phenotype[back_cnt_2 as usize][mid_cnt_2 as usize][front_cnt_2 as usize];
                
                //Choose pessimistically ie. select lesser of the two move probability 
                let move_prb= if move_prb_p1 > move_prb_p2 {move_prb_p1} else {move_prb_p2};
                
                //for Theory
                // let move_prb = ((SOPSepEnvironment::theory_gene_probability()[move_prb_p1 as usize] as f32 / 1000 as f32) * (SOPSepEnvironment::theory_gene_probability()[move_prb_p2 as usize] as f32 / 1000 as f32) * 1000 as f32) as u16;
                
                // if SOPSepEnvironment::move_frng().u16(1_u16..=1000) <= move_prb
                if SOPSepEnvironment::move_frng().u16(1_u16..=1000) <= SOPSepEnvironment::gene_probability()[move_prb as usize]
                {
                    self.move_particle_to(par_idx, move_dir);
                }
            }
            2 => {
                // move
                // Get the neighborhood configuration
                let (back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);
                // Move basis probability given by the genome for moving for given configuration
                //only for theory
                // if self.check_connectivity(par_idx, move_dir) {
                    let move_prb = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize];
                    if SOPSepEnvironment::move_frng().u16(1_u16..=1000) <= SOPSepEnvironment::gene_probability()[move_prb as usize]
                    {
                        self.move_particle_to(par_idx, move_dir);
                    }
                // }
            }
            _ => {}
        }
    }

    /*
     * Evaluate the measure of resultant configuration of SOPS grid
     * #. of total edges between every particle + #. of edges between particles of same color
     */
    pub fn evaluate_fitness(&self) -> f32 {
        let mut clr_edges = [0_u32; 3];
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            let neigbor_edges = self.get_neighbors_cnt(particle.x, particle.y, 0);
            clr_edges[(particle.state-1) as usize] += neigbor_edges.1 as u32;
            sum + neigbor_edges.0 as u32
        });
        
        let edge_cnt = ((edges as f32) / 2.0, (clr_edges.iter().sum::<u32>() as f32) / (2.0 * (clr_edges.len() as f32)));
        
        let c1 =  edge_cnt.0 / self.max_fitness_c1;
        let c2 = edge_cnt.1 / self.max_fitness_c2;

        self.w1 * c1 + self.w2 * c2
    }

    /*
     * Move a single particle at a time for 'sim_duration = fn(#. of particles)' times
     *  */
    pub fn simulate(&mut self, take_snaps: bool) -> f32 {
        for step in 0..self.sim_duration {
            self.move_particles(1 as usize);
            if take_snaps && (step == (self.participants.len() as u64) || step == (self.participants.len() as u64).pow(2) || step == (self.participants.len() as u64).pow(3) || step == (self.participants.len() as u64).pow(3)*2|| step == (self.participants.len() as u64).pow(3)*3|| step == (self.participants.len() as u64).pow(3)*4|| step == (self.participants.len() as u64).pow(3)*5) {
                println!("Step {}", step);
                self.print_grid();
                let fitness = self.evaluate_fitness();
                println!("Fitness: {}", fitness);
            }
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness;
        fitness
    }

    // pub fn get_max_fitness(&self) -> (f32, f32) {
    //     (self.max_fitness_c1, self.max_fitness_c1)
    // }

    pub fn get_participant_cnt(&self) -> [u16; 3] {
        let mut clr_particles = [0_u16; 3];
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                if self.grid[i][j] != 0 && self.grid[i][j] != 4 {
                    clr_particles[(self.grid[i][j]-1) as usize] += 1
                }
            }
        }
        clr_particles
    }
}
