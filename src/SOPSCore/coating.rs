use super::Particle;

use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};
use std::usize;
use std::collections::HashMap;
use std::collections::VecDeque;


/*
 * Main Class for the Separation Behaviour Experiments on SOPS grid
 * Defines how the genome is interpreted and how each transition of
 * particles is derived from the the genome. Also provides final SOPS
 * grid evaluations to assess the fitness score of the genome
 *  */
pub struct SOPSCoatEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [[[u8; 10]; 6]; 10],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    arena_layers: u16,
    object_layers: u16,
    coat_layers: u16,
    granularity: u8,
    lookup_dim_idx: HashMap<(u8, u8, u8), u8>,
    // obj_cen_loc: [usize; 2],
    w1: f32,
    w2: f32,
    grid_distances: HashMap<[usize; 2], u16>
}


impl SOPSCoatEnvironment {
    const EMPTY: u8 = 0;
    const PARTICLE: u8 = 1;
    // const CNCTPARTICLE: u8 = 2;
    const OBJECT: u8 = 2;
    const BOUNDARY: u8 = 7;

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
     * Uses BFS to calculate the shortest distance from location to nearest obj
     */

     fn get_obj_dist(&mut self, start: (usize, usize)) -> u16 {
        let mut visited: HashMap<[usize; 2], bool> = HashMap::new();
        let mut buffer: VecDeque<[usize; 3]> = VecDeque::new();

        buffer.push_back([start.0,start.1, 0]);

        while let Some(curr_loc) = buffer.pop_front() {
            for idx in 0..6 {
                let new_i = (curr_loc[0] as i32 + SOPSCoatEnvironment::directions()[idx].0) as usize;
                let new_j = (curr_loc[1] as i32 + SOPSCoatEnvironment::directions()[idx].1) as usize;
                let depth = curr_loc[2];
                if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                    match visited.get(&[new_i, new_j]) {
                        Some(_exists) => {}
                        None => {
                            if self.grid[new_i][new_j] == SOPSCoatEnvironment::OBJECT {
                                return (depth+1) as u16;
                            }
                            else if self.grid[new_i][new_j] == SOPSCoatEnvironment::EMPTY || self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE {
                                buffer.push_back([new_i, new_j, depth + 1]);
                                visited.insert([new_i, new_j], true);
                            }
                        },
                    }
                }
            }
        }
        return 0;
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
     
     // handcrafted configurations
     /*
    pub fn init_sops_env(genome: &[[[u8; 10]; 6]; 10], arena_layers: u16, object_layers: u16, coat_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32, grid_distances: Option<HashMap<[usize; 2], u16>>) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];

        let obj_locs: [(u8, u8); 150] = [(14, 16), (14, 17), (14, 18), (14, 19), (14, 20), (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), (15, 21), (15, 22), (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23), (17, 24), (17, 25), (17, 26), (17, 27), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26), (18, 27), (18, 28), (19, 19), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24), (19, 25), (19, 26), (19, 27), (19, 28), (20, 19), (20, 20), (20, 21), (20, 22), (20, 23), (21, 16), (21, 17), (21, 18), (21, 19), (21, 20), (21, 21), (21, 22), (21, 23), (21, 24), (21, 25), (22, 16), (22, 17), (22, 18), (22, 19), (22, 20), (22, 21), (22, 22), (22, 23), (22, 24), (22, 25), (22, 26), (22, 27), (22, 28), (23, 17), (23, 18), (23, 19), (23, 20), (23, 21), (23, 22), (23, 23), (23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29), (23, 30), (24, 23), (24, 24), (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), (24, 30), (24, 31), (24, 32), (24, 33), (25, 25), (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), (25, 31), (25, 32), (25, 33), (25, 34), (26, 27), (26, 28), (26, 29), (26, 30), (26, 31), (26, 32), (26, 33), (26, 34), (27, 27), (27, 28), (27, 29), (27, 30), (27, 31), (27, 32), (27, 33), (27, 34), (28, 24), (28, 25), (28, 26), (28, 27), (28, 28), (28, 29), (28, 30), (28, 31), (28, 32), (28, 33), (29, 24), (29, 25), (29, 26), (29, 27), (29, 28), (29, 29), (29, 30), (29, 31), (30, 25), (30, 26), (30, 27), (30, 28), (30, 29)];

        let num_particles = 500;

        let mut grid_rng = SOPSCoatEnvironment::seed_rng(seed);

        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = SOPSCoatEnvironment::BOUNDARY;
                grid[(i+arena_layers+j) as usize][i as usize] = SOPSCoatEnvironment::BOUNDARY;
                j +=1;
            }
        }
        
        for i in 0..grid_size {
            for j in 0..grid_size {
                if obj_locs.iter().any(|&x| x.0 == (i as u8) && x.1== (j as u8)) {
                    grid[i][j] = SOPSCoatEnvironment::OBJECT;
                }
            }
        }

        //init grid and particles
        while participants.len() < num_particles {
            let i = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            let j = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    state: 0
                });
                grid[i][j] = SOPSCoatEnvironment::PARTICLE;
            }   
        }

        // TODO: Make this a static const variable
        // Mapping table for various intra group(F/M/B) configurations -> index in genome's dimension
        // intra group(F/M/B) configurations ie. particle_cnt, object_cnt, all_possible_cnt(static position cnt in F/M/B — (3/2/3))
        let lookup_dim_idx: HashMap<(u8, u8, u8), u8> = ([
            ((0,0,2), 0), // (0,0,2)
            ((0,1,2), 1), //  
            ((0,2,2), 2), // 
            ((1,1,2), 3), // 
            ((1,0,2), 4), // 
            ((2,0,2), 5), // 

            ((0,0,3), 0), // (0,0,3)
            ((0,1,3), 1), // 
            ((0,2,3), 2), // 
            ((0,3,3), 3), // 
            ((2,1,3), 4), // 
            ((1,1,3), 5), // 
            ((1,2,3), 6), // 
            ((1,0,3), 7), // 
            ((2,0,3), 8), // 
            ((3,0,3), 9), // 
        ]).into();

        SOPSCoatEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            arena_layers,
            object_layers,
            coat_layers,
            granularity,
            lookup_dim_idx,
            w1,
            w2,
            grid_distances: grid_distances.unwrap_or(HashMap::new())
        }
    }
     */

    //  /*
    pub fn init_sops_env(genome: &[[[u8; 10]; 6]; 10], arena_layers: u16, object_layers: u16, coat_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32, grid_distances: Option<HashMap<[usize; 2], u16>>) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];

        let total_coat = object_layers + coat_layers;
        let num_particles = (6*total_coat*(1+total_coat)/2) - (6*object_layers*(1+object_layers)/2);
        // println!("Particles: {}", num_particles);

        // let num_empty = (6*arena_layers*(1+arena_layers)/2) - (6*object_layers*(1+object_layers)/2);
        // println!("Empty space: {}", num_empty);

        // println!("Density {}", (num_particles as f32) / (num_empty as f32));

        let total_coat_size = (total_coat*2 + 1) as usize;

        
        let object_size = (object_layers*2 + 1) as usize;

        let mut grid_rng = SOPSCoatEnvironment::seed_rng(seed);

        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = SOPSCoatEnvironment::BOUNDARY;
                grid[(i+arena_layers+j) as usize][i as usize] = SOPSCoatEnvironment::BOUNDARY;
                j +=1;
            }
        }
        
        //Try to randomly place the object(s)
        let mut obj_loc = vec![0; 2];
        
        loop {
            // place the object in center
            let x = (arena_layers - object_layers) as usize;
            let y = (arena_layers - object_layers) as usize;
            if (y+total_coat_size) < grid_size && (x+total_coat_size) < grid_size {
                if grid[x][y] != SOPSCoatEnvironment::BOUNDARY && grid[x][y+total_coat_size] != SOPSCoatEnvironment::BOUNDARY && grid[x+total_coat_size][y] != SOPSCoatEnvironment::BOUNDARY && grid[x+total_coat_size][y+total_coat_size] != SOPSCoatEnvironment::BOUNDARY {
                    obj_loc[0] = x;
                    obj_loc[1] = y;
                    break;
                }   
            }
        }

        //Mark object locations
        for i in 0..object_size {
            let mut j = i;
            while j < (i+(object_layers as usize)+1) && j < object_size {
                grid[(obj_loc[0] + (coat_layers as usize) + i) as usize][(obj_loc[1] + (coat_layers as usize) + j) as usize] = SOPSCoatEnvironment::OBJECT;
                grid[(obj_loc[0] + (coat_layers as usize) + j) as usize][(obj_loc[1] + (coat_layers as usize) + i) as usize] = SOPSCoatEnvironment::OBJECT;
                j +=1;
            }
        }

        //init grid and particles
        while participants.len() < num_particles.into() {
            let i = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            let j = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    state: 0
                });
                grid[i][j] = SOPSCoatEnvironment::PARTICLE;
            }   
        }

        // Mapping table for various intra group(F/M/B) configurations -> index in genome's dimension
        // intra group(F/M/B) configurations ie. particle_cnt, object_cnt, all_possible_cnt(static position cnt in F/M/B — (3/2/3))
        let lookup_dim_idx: HashMap<(u8, u8, u8), u8> = ([
            ((0,0,2), 0), // (0,0,2)
            ((0,1,2), 1), //  
            ((0,2,2), 2), // 
            ((1,1,2), 3), // 
            ((1,0,2), 4), // 
            ((2,0,2), 5), // 

            ((0,0,3), 0), // (0,0,3)
            ((0,1,3), 1), // 
            ((0,2,3), 2), // 
            ((0,3,3), 3), // 
            ((2,1,3), 4), // 
            ((1,1,3), 5), // 
            ((1,2,3), 6), // 
            ((1,0,3), 7), // 
            ((2,0,3), 8), // 
            ((3,0,3), 9), // 
        ]).into();

        SOPSCoatEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            arena_layers,
            object_layers,
            coat_layers,
            granularity,
            lookup_dim_idx,
            w1,
            w2,
            grid_distances: grid_distances.unwrap_or(HashMap::new())
        }
    }
    // */

    pub fn save_distance_grid(&mut self) -> HashMap<[usize; 2], u16> {
        //Calculate distance metrics from each empty position to nearest obj
        let mut distances: HashMap<[usize; 2], u16> = HashMap::new();

        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                if self.grid[i][j] == SOPSCoatEnvironment::EMPTY || self.grid[i][j] == SOPSCoatEnvironment::PARTICLE {
                    distances.insert([i, j], self.get_obj_dist((i,j)));
                }
            }
        }

        self.grid_distances = distances.clone();

        return distances;
    }

    pub fn print_dist_grid(&self) {
        println!("DIST grid");
        
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                if self.grid[i][j] == SOPSCoatEnvironment::EMPTY || self.grid[i][j] == SOPSCoatEnvironment::PARTICLE {
                    match self.grid_distances.get(&[i, j]) {
                            Some(dist) => { print!(" {} ", dist); }
                            None => { print!(" - "); },
                        }
                }
                else {
                    print!(" X ");
                }
            }
            println!("");
        }
    }

    pub fn get_min_total_dist(&self) -> u32 {
        let num_particles = self.participants.len();
        let mut dist_vec: Vec<u16> = Vec::new();
        for (_, dist) in self.grid_distances.iter() {
            dist_vec.push(*dist);
        }

        dist_vec.sort();

        let mut min_dist_sum: u32 = 0;
        for idx in 0..num_particles {
            min_dist_sum += dist_vec[idx] as u32;
        }

        return min_dist_sum;
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
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        
        self.grid[particle.x as usize][particle.y as usize] -= SOPSCoatEnvironment::PARTICLE;
        
        self.grid[new_i][new_j] += SOPSCoatEnvironment::PARTICLE;

        let mut particle_mut = &mut self.participants[particle_idx];
        particle_mut.x = new_i as u8;
        particle_mut.y = new_j as u8;
        return true;
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
     * Func to calculate a particle's extended neighbor count and connected state
     *  */
    fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8) {
        let mut back_cnt: u8 = 0;
        let mut back_obj_cnt: u8 = 0;
        let mut mid_cnt: u8 = 0;
        let mut mid_obj_cnt: u8 = 0;
        let mut front_cnt: u8 = 0;
        let mut front_obj_cnt: u8 = 0;
        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSCoatEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSCoatEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == move_i) & (new_j == move_j)) {
                // print!("{}",idx);
                seen_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] == SOPSCoatEnvironment::OBJECT {
                    back_obj_cnt += 1;
                }
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE {
                    back_cnt += 1;
                }
            }
        }
        // print!("\t");
        // Neighborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSCoatEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSCoatEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                // print!("{}",idx);
                let mut position_type = SOPSCoatEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSCoatEnvironment::MID;
                    }
                    None => {},
                }
                if self.grid[new_i][new_j] == SOPSCoatEnvironment::OBJECT {
                    match position_type {
                        SOPSCoatEnvironment::FRONT => {
                            front_obj_cnt += 1;
                        }
                        SOPSCoatEnvironment::MID => {
                            mid_obj_cnt += 1;
                            back_obj_cnt -= 1;
                        }
                        _ => todo!()
                    }
                }
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE {
                    match position_type {
                        SOPSCoatEnvironment::FRONT => {
                            front_cnt += 1;
                        }
                        SOPSCoatEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                        }
                        _ => todo!()
                    }
                }
            }
        }
        // print!("\t");
        let back_idx: u8 = self.get_dim_idx(back_cnt, back_obj_cnt, 3);
        let mid_idx: u8 = self.get_dim_idx(mid_cnt, mid_obj_cnt, 2);
        let front_idx: u8 = self.get_dim_idx(front_cnt, front_obj_cnt, 3);
        (back_idx.clamp(0, 9), mid_idx.clamp(0, 5), front_idx.clamp(0, 9))
    }

    /*
     * Func to check if the proposed move is possible or not for a particle
     *  */
    fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // Move particle if movement is within grid array's bound
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            if self.grid[new_i][new_j] == SOPSCoatEnvironment::EMPTY {
                return true;
            }
            else {
                return false;
            }
        } else {
            return false;
        }
    }

    /*
     * Func to move 'n' particles in random directions in the SOPS grid
     */
    fn move_particles(&mut self, cnt: usize) {
        
        // Choose a random particle for movement
        let par_idx = SOPSCoatEnvironment::move_frng().usize(..self.participants.len());
        
        // Choose a direction at random (out of the 6)
        let move_dir = SOPSCoatEnvironment::directions()
            [SOPSCoatEnvironment::move_frng().usize(..SOPSCoatEnvironment::directions().len())];
            
        if self.particle_move_possible(par_idx, move_dir) {
            // move
            // Get the neighborhood configuration
            let (back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);
            // Move basis probability given by the genome for moving for given configuration
            let move_prb = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize];
            if SOPSCoatEnvironment::move_frng().u16(1_u16..=1000) <= SOPSCoatEnvironment::gene_probability()[move_prb as usize]
            {
                self.move_particle_to(par_idx, move_dir);
            }
        }
    }

    /*
     * Evaluate the measure of resultant configuration of SOPS grid
     * #. of total edges between every particle + #. of edges between particles of same color
     */
    pub fn evaluate_fitness(&self) -> f32 {
        let min_total_dist = self.get_min_total_dist();
        let mut par_total_dist: u32 = 0;
        self.participants.iter().for_each(|par| {
            match self.grid_distances.get(&[par.x as usize, par.y as usize]) {
                Some(dist) => {
                    par_total_dist += *dist as u32;
                }
                None => {},
            }
        });
        // println!("Min Dist: {}   Particles Dist: {}", min_total_dist,  par_total_dist);
        return (min_total_dist as f32)/ (par_total_dist as f32);
    }

    /*
     * Move a single particle at a time for 'sim_duration = fn(#. of particles)' times
     *  */
    pub fn simulate(&mut self, take_snaps: bool) -> f32 {
        for step in 0..self.sim_duration {
            self.move_particles(1 as usize);
            if take_snaps && (step == (self.participants.len() as u64) || step == (self.participants.len() as u64).pow(2) || step == (self.participants.len() as u64).pow(3) || step == (self.participants.len() as u64).pow(2)*20 || step == (self.participants.len() as u64).pow(2)*40 || step == (self.participants.len() as u64).pow(2)*60|| step == (self.participants.len() as u64).pow(2)*70) {
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
