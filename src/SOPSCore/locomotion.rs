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
pub struct SOPSLocoEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    light: Vec<(u8,u8)>,
    orientation: u8,
    phenotype: [[[[[u8; 10]; 6]; 10]; 2]; 2],
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


impl SOPSLocoEnvironment {
    const EMPTY: u8 = 0;
    const PARTICLE: u8 = 1;
    const BOUNDARY: u8 = 2;

    const BACK: u8 = 0;
    const MID: u8 = 1;
    const FRONT: u8 = 2;

    // const W1: f32 = 0.65; //weight for aggregation
    // const W2: f32 = 0.35; //weight for distance

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

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    /*
     * Initialize a SOPS grid and place particles based on particle layer and arena layer count
     * Parameters Particle layers and Arena layers refer to the complete hexagonal lattice layers
     * of the SOPS grid and this also defines the total density of particles in the arena.
     * Calculates Max edge count possible for all the particles irrespective of the color
     * Calculates Max edge count possible for all the particles of the same color
     * Also accept the weights for Agg and Light components
     * NOTE: Use the Same random Seed value to get the same random init config
     *  */
    pub fn init_sops_env(genome: &[[[[[u8; 10]; 6 ]; 10]; 2]; 2], arena_layers: u16, particle_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;

        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let num_particles = 6*particle_layers*(1+particle_layers)/2;


        // stores coordinates of particle sensing light
        let mut light: Vec<(u8, u8)> = vec![(0,0); (arena_layers + 1) as usize]; 

        let orientation = rand::thread_rng().gen_range(1..3);
        //let orientation = 3;

        // reflects direction of side light emenates from
        let mut axis_direction = (0, 0); 

        if orientation == 1{
            light[0] = (arena_layers as u8, 0);
            axis_direction = (1, 1);
        }
        else if orientation == 2 {
            light[0] = (0, arena_layers as u8);
            axis_direction = (1, 1);
        }
        else if orientation == 3 {
            light[0] = (0, 0);
            axis_direction = (0, 1);
        }

        // set to values along axis direction
        for i in 1..arena_layers {
            light[i as usize].0 = light[(i-1) as usize].0 + axis_direction.0;
            light[i as usize].1 = light[(i-1) as usize].1 + axis_direction.1;
        }
        
        
        // No. of edges in the aggregated config with one less particle than aggregation behavior
        let agg_edge_cnt: f32 = (3*(3*particle_layers.pow(2)+particle_layers-1)).into();
        let agg_avg_distance: f32 = (grid_size as f32 - ((-1 as f32 + f32::sqrt(1 as f32 + 4 as f32 * (num_particles as f32 - 1 as f32)/3 as f32))/2 as f32)).into();
        
        let mut grid_rng = SOPSLocoEnvironment::seed_rng(seed);
        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i+arena_layers+j < (grid_size as u16) {
                grid[i as usize][(i+arena_layers+j) as usize] = SOPSLocoEnvironment::BOUNDARY;
                grid[(i+arena_layers+j) as usize][i as usize] = SOPSLocoEnvironment::BOUNDARY;
                j +=1;
            }
        }
        
        //init grid and particles
        while participants.len() < num_particles.into() {
            let i = grid_rng.sample(&SOPSLocoEnvironment::grid_rng(0,grid_size));
            let j = grid_rng.sample(&SOPSLocoEnvironment::grid_rng(0,grid_size));
            if grid[i][j] == 0 {
                participants.push(Particle {
                    x: i as u8,
                    y: j as u8,
                    color: 0
                });
                grid[i][j] = SOPSLocoEnvironment::PARTICLE;

                // initialize values for light
                if orientation == 1 {
                    if i as u8 >= arena_layers as u8 {
                        if j as u8 >= light[i - arena_layers as usize].1 {
                            light[i - arena_layers as usize] = (i as u8, j as u8);
                        }
                    }
        
                }
                else if orientation == 2 {
                    if j as u8 >= arena_layers as u8 {
                        if i as u8 >= light[j - arena_layers as usize].0 {
                            light[j - arena_layers as usize] = (i as u8, j as u8);
                        }
                    }
                }
                else if orientation == 3 {
                    if i <= j {
                        if j as u8 >= light[j - i].1 {
                            light[j - i] = (i as u8, j as u8);
                        }
                    }
                }
            }   
        }
        
        // TODO: Make this a static const variable
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

        SOPSLocoEnvironment {
            grid,
            participants,
            light,
            orientation,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3)*3,
            fitness_val: 0.0,
            size: grid_size,
            max_fitness_c1: agg_edge_cnt,
            max_fitness_c2: agg_avg_distance,
            arena_layers,
            particle_layers,
            granularity,
            lookup_dim_idx,
            w1,
            w2
        }
    }

    /*
     * Func to get if particle senses light
     */
    fn sensing_light(&self, x: u8, y: u8) -> u8 {
        if self.orientation == 1 {
            if x >= self.arena_layers as u8 {
                if y >= self.light[(x - self.arena_layers as u8) as usize].1 {
                    return 1;
                }
            }

        }
        else if self.orientation == 2 {
            if y >= self.arena_layers as u8 {
                if x >= self.light[(y - self.arena_layers as u8) as usize].0 {
                    return 1;
                }
            }

        }
        else if self.orientation == 3 {
            if x <= y {
                if y >= self.light[(y - x) as usize].1 {
                    return 1;
                }
            }
        }
        return 0;
    }

    /*fn update_light(&mut self, current_x: u8, current_y: u8, future_x: u8, future_y: u8) {
        if self.orientation == 1 && self.sensing_light(current_x, current_y) == 1 {
            // same beam
            if current_x - future_x == 0 {
                self.light[current_x as usize] = (future_x, future_y);
            }
            else {
                // update closest particle sensing light
                for j in current_y..0 {
                    if self.grid[current_x as usize][j as usize] == SOPSLocoEnvironment::PARTICLE {
                        self.light[current_x as usize].1 = j;
                        break;
                    }
                }
                // adjust future light value
                if self.sensing_light(future_x, future_y) == 1 {
                    self.light[future_x as usize] = (future_x, future_y);
                }
            }
        }
    }*/

    pub fn print_grid(&self) {
        println!("Locomotion grid");
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                print!(" {} ", self.grid[i][j]);
            }
            println!("");
        }
    }

    /* 
     * Func to calculate a particle's neighbor count both 
     *  */
     fn get_neighbors_cnt(&self, i: u8, j: u8) -> u8 {
        let mut cnt = 0;
        for idx in 0..6 {
            let new_i = (i as i32 + SOPSLocoEnvironment::directions()[idx].0) as usize;
            let new_j = (j as i32 + SOPSLocoEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                if self.grid[new_i][new_j] == 1 {
                    cnt += 1;
                }
            }
        }
        cnt
    }
    
    /* Copied from mod.rs
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
     fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let mut particle = &mut self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;

        // check if sensing light
        let mut sensing_light = 0;
        let mut future_sensing_light = 0;
        if self.orientation == 1 {
            if particle.x >= self.arena_layers as u8 {
                if particle.y >= self.light[(particle.x - self.arena_layers as u8) as usize].1 {
                    sensing_light = 1;
                }
            }
            if new_i as u8 >= self.arena_layers as u8 {
                if new_j as u8 >= self.light[(new_i as u8 - self.arena_layers as u8) as usize].1 {
                    future_sensing_light = 1;
                }
            }

        }
        else if self.orientation == 2 {
            if particle.y >= self.arena_layers as u8 {
                if particle.x >= self.light[(particle.y - self.arena_layers as u8) as usize].0 {
                    sensing_light = 1;
                }
            }
            if new_j as u8 >= self.arena_layers as u8 {
                if new_i as u8 >= self.light[(new_j as u8 - self.arena_layers as u8) as usize].0 {
                    future_sensing_light = 1;
                }
            }

        }
        else if self.orientation == 3 {
            if particle.x <= particle.y {
                if particle.y >= self.light[(particle.y - particle.x) as usize].1 {
                    sensing_light = 1;
                }
            }
            if new_i <= new_j {
                if new_j as u8 >= self.light[(new_j - new_i) as usize].1 {
                    future_sensing_light = 1;
                }
            }
        }

        //self.update_light(particle.x, particle.y, new_i as u8, new_j as u8);
        if self.orientation == 1 && sensing_light == 1 {
            // same beam
            if particle.x == new_i as u8{
                self.light[particle.x as usize - self.arena_layers as usize] = (new_i as u8, new_j as u8);
            }
            else {
                // update closest particle sensing light
                for j in particle.y..0 {
                    if self.grid[particle.x as usize - self.arena_layers as usize][j as usize] == SOPSLocoEnvironment::PARTICLE {
                        self.light[particle.x as usize - self.arena_layers as usize].1 = j;
                        break;
                    }
                }
                // adjust future light value
                if future_sensing_light == 1 {
                    self.light[new_i - self.arena_layers as usize] = (new_i as u8, new_j as u8);
                }
            }
        }
        else if self.orientation == 2 && sensing_light == 1 {
            // same beam
            if particle.y == new_j as u8{
                self.light[particle.y as usize - self.arena_layers as usize] = (new_i as u8, new_j as u8);
            }
            else {
                // update closest particle sensing light
                for i in particle.x..0 {
                    if self.grid[i as usize][particle.y as usize - self.arena_layers as usize] == SOPSLocoEnvironment::PARTICLE {
                        self.light[particle.y as usize - self.arena_layers as usize].0 = i;
                        break;
                    }
                }
                // adjust future light value
                if future_sensing_light == 1 {
                    self.light[new_j - self.arena_layers as usize] = (new_i as u8, new_j as u8);
                }
            }
        }
        else if self.orientation == 3 && sensing_light == 1 {
            // future location in light's path
            if (new_i <= new_j){
                // same beam
                if (particle.y - particle.x) as u8 == (new_j - new_i) as u8{
                    self.light[(particle.y - particle.x) as usize] = (new_i as u8, new_j as u8);
                }
                else {
                    // update closest particle sensing light
                    for i in 0..particle.x {
                        if self.grid[(particle.x - i) as usize][(particle.y - i) as usize] == SOPSLocoEnvironment::PARTICLE {
                            self.light[(particle.y - particle.x) as usize] = (particle.x - i, particle.y - i);
                            break;
                        }
                    }

                    // adjust future light value
                    if future_sensing_light == 1 {
                        self.light[(new_j - new_i) as usize] = (new_i as u8, new_j as u8);
                    }
                }
            }
            else {
                // particle moving out of light's path
                // update closest particle sensing light
                for i in 0..particle.x {
                    if self.grid[(particle.x - i) as usize][(particle.y - i) as usize] == SOPSLocoEnvironment::PARTICLE {
                        self.light[(particle.y - particle.x) as usize] = (particle.x - i, particle.y - i);
                        break;
                    }
                }
            }
        }

        // move the particle
        self.grid[particle.x as usize][particle.y as usize] = SOPSLocoEnvironment::EMPTY;
        self.grid[new_i][new_j] = SOPSLocoEnvironment::PARTICLE;
        particle.x = new_i as u8;
        particle.y = new_j as u8;

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
     * Func to calculate a particle's extended neighbor count, includes if sensing light
     *  */
     fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8, u8, u8) {
        let mut back_cnt: u8 = 0;
        let mut back_light_cnt: u8 = 0;
        let mut mid_cnt: u8 = 0;
        let mut mid_light_cnt: u8 = 0;
        let mut front_cnt: u8 = 0;
        let mut front_light_cnt: u8 = 0;
        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSLocoEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSLocoEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                seen_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] != SOPSLocoEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSLocoEnvironment::BOUNDARY  {
                    back_cnt += 1;
                    if self.orientation == 1 && new_i as u16 >= self.arena_layers {
                        if new_j == self.light[new_i - self.arena_layers as usize].1 as usize {
                            back_light_cnt += 1;
                        }
                    }
                    else if self.orientation == 2 && new_j as u16 >= self.arena_layers {
                        if new_i == self.light[new_j - self.arena_layers as usize].0 as usize {
                            back_light_cnt += 1;
                        }
                    }
                    else if self.orientation == 3 && new_i <= new_j {
                        if (new_i as u8, new_j as u8) == self.light[(new_j - new_i) as usize] {
                            back_light_cnt += 1;
                        }
                    }
                }
            }
        }
        // Neighborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSLocoEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSLocoEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                let mut position_type = SOPSLocoEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSLocoEnvironment::MID;
                    }
                    None => {},
                }
                if self.grid[new_i][new_j] != SOPSLocoEnvironment::EMPTY && self.grid[new_i][new_j] != SOPSLocoEnvironment::BOUNDARY {
                    match position_type {
                        SOPSLocoEnvironment::FRONT => {
                            front_cnt += 1;
                            if self.orientation == 1 && new_i as u16 >= self.arena_layers {
                                if new_j == self.light[new_i - self.arena_layers as usize].1 as usize {
                                    front_light_cnt += 1;
                                }
                            }
                            else if self.orientation == 2 && new_j as u16 >= self.arena_layers {
                                if new_i == self.light[new_j - self.arena_layers as usize].0 as usize {
                                    front_light_cnt += 1;
                                }
                            }
                            else if self.orientation == 3 && new_i <= new_j {
                                if (new_i as u8, new_j as u8) == self.light[(new_j - new_i) as usize] {
                                    front_light_cnt += 1;
                                }
                            }
                        }
                        SOPSLocoEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                            if self.orientation == 1 && new_i as u16 >= self.arena_layers {
                                if new_j == self.light[new_i - self.arena_layers as usize].1 as usize {
                                    mid_light_cnt += 1;
                                    back_light_cnt -= 1;
                                }
                            }
                            else if self.orientation == 2 && new_j as u16 >= self.arena_layers {
                                if new_i == self.light[new_j - self.arena_layers as usize].0 as usize {
                                    mid_light_cnt += 1;
                                    back_light_cnt -= 1;
                                }
                            }
                            else if self.orientation == 3 && new_i <= new_j {
                                if (new_i as u8, new_j as u8) == self.light[(new_j - new_i) as usize] {
                                    mid_light_cnt += 1;
                                    back_light_cnt -= 1;
                                }
                            }
                        }
                        _ => todo!()
                    }
                }
            }
        }
        let back_idx: u8 = self.get_dim_idx(back_cnt, back_light_cnt, 3);
        let mid_idx: u8 = self.get_dim_idx(mid_cnt, mid_light_cnt, 2);
        let front_idx: u8 = self.get_dim_idx(front_cnt, front_light_cnt, 3);
        let future_idx: u8 = self.sensing_light(move_i as u8, move_j as u8);
        let current_idx: u8 = self.sensing_light(particle.x, particle.y);
        // TODO: Remove this hardcoding of the values. Should come from genome's dimenions
        (current_idx.clamp(0, 1), future_idx.clamp(0, 1), back_idx.clamp(0, 9), mid_idx.clamp(0, 5), front_idx.clamp(0, 9))
    }

    /* Copied from mod.rs
     * Func to check if the proposed move is possible or not for a particle
     *  */
     fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // Move particle if movement is within bound
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // check to see if move is valid ie. within bounds and not in an already occupied location
            if self.grid[new_i][new_j] == SOPSLocoEnvironment::PARTICLE || self.grid[new_i][new_j] == SOPSLocoEnvironment::BOUNDARY {
                return false;
            } else {
                // can move the particle
                return true;
            }
        } else {
            return false;
        }
    }

    /*
     * Func to check if the proposed move is possible or not for a particle
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
        // let mut par_moves: Vec<(usize, (i32, i32))> = Vec::new();

        //  for _ in 0..cnt {
            // Choose a random particle for movement
            let par_idx = SOPSLocoEnvironment::rng().sample(&self.unfrm_par());
            // Choose a direction at random (out of the 6)
            let move_dir = SOPSLocoEnvironment::directions()
                        [SOPSLocoEnvironment::rng().sample(&SOPSLocoEnvironment::unfrm_dir())];
            
            if self.particle_move_possible(par_idx, move_dir) {
                // Get the neighborhood configuration
                let (current_light, future_light, back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);
                // Move basis probability given by the genome for moving for given configuration
                // TODO: Change this simply using a (0, granularity) for RNG and compare values basis that
                let move_prb: f64 =
                    self.phenotype[current_light as usize][future_light as usize][back_cnt as usize][mid_cnt as usize][front_cnt as usize] as f64 / (self.granularity as f64);
                if SOPSLocoEnvironment::move_frng().u64(1_u64..=10000)
                    <= (move_prb * 10000.0) as u64
                {
                    self.move_particle_to(par_idx, move_dir);
                }
            }
                
    }
    

    /*
     * Evaluate the measure of resultant configuration of Locomotion grid
     * #. of total edges between every particle + avg. distance of particles
     */
    pub fn evaluate_fitness(&self) -> f32 {
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            sum + self.get_neighbors_cnt(particle.x, particle.y) as u32
        });

        let mut distance = 0;
        if self.orientation == 1 {
            distance = self.participants.iter().fold(0, |sum: u32, particle| {
                sum + particle.y as u32
            });
        }
        else if self.orientation == 2 {
            distance = self.participants.iter().fold(0, |sum: u32, particle| {
                sum + particle.x as u32
            });
        }
        else if self.orientation == 3 {
            distance = self.participants.iter().fold(0, |sum: u32, particle| {
                sum + particle.y as u32
            });
        }

        let c1 = (edges / 2) as f32 / self.max_fitness_c1; //aggregation
        let c2 = (distance / self.get_participant_cnt() as u32) as f32 / self.max_fitness_c2; //avg distance
        
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
                // Check to see if swaps and other motion is working correctly by checking total #. of particles
                // println!("No. of Participants {:?}", self.get_participant_cnt());
                // let particles_cnt = self.get_participant_cnt();
                // if particles_cnt.iter().any(|&x| x != (self.participants.len() as u16/3)) {
                //     panic!("Something is wrong");
                // }
                let fitness = self.evaluate_fitness();
                println!("Fitness: {}", fitness);
            }
        }
        let fitness = self.evaluate_fitness();
        self.fitness_val = fitness;
        fitness
    }

    pub fn get_participant_cnt(&self) -> usize {
        self.participants.len()
    }
    
}