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
    phenotype: [[[u8; 11]; 7]; 11],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    arena_layers: u16,
    object_layers: u16,
    coat_layers: u16,
    granularity: u8,
    max_inner: u16,
    max_outer: u16,
    lookup_dim_idx: HashMap<(u8, u8, u8), u8>,
    connected_par_loc: HashMap<(u8, u8), bool>,
    obj_cen_loc: [usize; 2],
    w1: f32,
    w2: f32,
}


impl SOPSCoatEnvironment {
    const EMPTY: u8 = 0;
    const PARTICLE: u8 = 1;
    // const CNCTPARTICLE: u8 = 2;
    const OBJECT: u8 = 2;
    const INCOAT: u8 = 3; // -> 4
    const COAT: u8 = 5; // -> 6
    const BOUNDARY: u8 = 7;

    const BACK: u8 = 0;
    const MID: u8 = 1;
    const FRONT: u8 = 2;

    // const W1: f32 = 0.65; //Inner coat weightage
    // const W2: f32 = 0.35; //Outer coat weightage

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
    pub fn init_sops_env(genome: &[[[u8; 11]; 7]; 11], arena_layers: u16, object_layers: u16, coat_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];

        let total_coat = object_layers + coat_layers;
        let num_particles = (6*total_coat*(1+total_coat)/2) - (6*object_layers*(1+object_layers)/2);

        let total_coat_size = (total_coat*2 + 1) as usize;

        let inner_coat_layers = object_layers + 1;
        let inner_coat_size = (inner_coat_layers*2 + 1) as usize;

        let num_inner_pos = (6*inner_coat_layers*(1+inner_coat_layers)/2) - (6*object_layers*(1+object_layers)/2);
        let num_outer_pos = num_particles - num_inner_pos;
        
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
            let x = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            let y = grid_rng.sample(&SOPSCoatEnvironment::grid_rng(0,grid_size));
            if (y+total_coat_size) < grid_size && (x+total_coat_size) < grid_size {
                if grid[x][y] != SOPSCoatEnvironment::BOUNDARY && grid[x][y+total_coat_size] != SOPSCoatEnvironment::BOUNDARY && grid[x+total_coat_size][y] != SOPSCoatEnvironment::BOUNDARY && grid[x+total_coat_size][y+total_coat_size] != SOPSCoatEnvironment::BOUNDARY {
                    obj_loc[0] = x;
                    obj_loc[1] = y;
                    break;
                }   
            }
        }

        //First mark the coat locations
        for i in 0..total_coat_size {
            let mut j = i;
            while j < (i+(total_coat as usize)+1) && j < total_coat_size {
                grid[(obj_loc[0] + i) as usize][(obj_loc[1] + j) as usize] = SOPSCoatEnvironment::COAT;
                grid[(obj_loc[0] + j) as usize][(obj_loc[1] + i) as usize] = SOPSCoatEnvironment::COAT;
                j +=1;
            }
        }

        //Then mark inner coat locations
        for i in 0..inner_coat_size {
            let mut j = i;
            while j < (i+(inner_coat_layers as usize)+1) && j < inner_coat_size {
                grid[(obj_loc[0] + ((coat_layers as usize)-1) + i) as usize][(obj_loc[1] + ((coat_layers as usize)-1) + j) as usize] = SOPSCoatEnvironment::INCOAT;
                grid[(obj_loc[0] + ((coat_layers as usize)-1) + j) as usize][(obj_loc[1] + ((coat_layers as usize)-1) + i) as usize] = SOPSCoatEnvironment::INCOAT;
                j +=1;
            }
        }

        //Then mark object locations
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

        // TODO: Make this a static const variable
        // Mapping table for various intra group(F/M/B) configurations -> index in genome's dimension
        // intra group(F/M/B) configurations ie. all_cnt, same_clr_cnt, all_possible_cnt(static position cnt in F/M/B — (3/2/3))
        let lookup_dim_idx: HashMap<(u8, u8, u8), u8> = ([
            ((0,0,2), 0),
            ((1,1,2), 1), //
            ((1,0,2), 2), 
            ((2,2,2), 3), //
            ((2,1,2), 4), //
            ((2,0,2), 5),
            ((0,0,3), 0),
            ((1,1,3), 1), //
            ((1,0,3), 2),
            ((2,2,3), 3), //
            ((2,1,3), 4), //
            ((2,0,3), 5),
            ((3,3,3), 6), //
            ((3,2,3), 7), //
            ((3,1,3), 8), //
            ((3,0,3), 9),
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
            max_inner: num_inner_pos,
            max_outer: num_outer_pos,
            granularity,
            lookup_dim_idx,
            w1,
            w2,
            connected_par_loc: HashMap::new(),
            obj_cen_loc: [(obj_loc[0]+total_coat as usize), (obj_loc[1]+total_coat as usize)],
        }
    }

    pub fn print_grid(&self) {
        println!("SOPS grid");
        // const EMPTY: u8 = 0;
        // const PARTICLE: u8 = 1;
        // const OBJECT: u8 = 2;
        // const INCOAT: u8 = 3; // -> 4
        // const COAT: u8 = 5; // -> 6
        // const BOUNDARY: u8 = 7;
        // CONNECTED_PAR = 9
        // for (key, _value) in &self.connected_par_loc {
        //     println!("{:?} ", key);
        // }
        for i in 0..self.grid.len() {
            for j in 0..self.grid[0].len() {
                if self.grid[i][j] == SOPSCoatEnvironment::COAT || self.grid[i][j] == SOPSCoatEnvironment::INCOAT {
                    print!(" 0 ");
                }
                else if self.grid[i][j] == SOPSCoatEnvironment::PARTICLE
                    || (self.grid[i][j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE
                    || (self.grid[i][j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE {
                    match self.connected_par_loc.get(&(i as u8, j as u8)) {
                        Some(_exists) => { print!(" 9 "); }
                        None => { print!(" 1 "); },
                    }
                }
                else {
                    print!(" {} ", self.grid[i][j]);
                }
            }
            println!("");
        }
    }

    /*
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32), front_cnt: u8, mid_cnt: u8) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        
        self.grid[particle.x as usize][particle.y as usize] -= SOPSCoatEnvironment::PARTICLE;
        
        // check if the front or mid has any connected particles or object itself to determine moving particle's state
        // let be_connected = vec![1,3,4,6,7,8,10].iter().any(|&x| front_cnt == x) || vec![1,3,4,6].iter().any(|&x| mid_cnt == x);

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
        let mut back_cnct_cnt: u8 = 0;
        let mut back_obj_seen: bool = false;
        let mut mid_cnt: u8 = 0;
        let mut mid_cnct_cnt: u8 = 0;
        let mut mid_obj_seen: bool = false;
        let mut front_cnt: u8 = 0;
        let mut front_cnct_cnt: u8 = 0;
        let mut front_obj_seen: bool = false;
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
                    back_obj_seen = true;
                }
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE {
                    match self.connected_par_loc.get(&(new_i as u8, new_j as u8)) {
                        Some(_exists) => {
                            back_cnt += 1;
                            back_cnct_cnt += 1;
                        }
                        None => { back_cnt += 1; },
                    }
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
                            front_obj_seen = true;
                        }
                        SOPSCoatEnvironment::MID => {
                            mid_obj_seen = true;
                            back_obj_seen = false;
                        }
                        _ => todo!()
                    }
                }
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE {
                    // check if the particle exists in connected particle list
                    match self.connected_par_loc.get(&(new_i as u8, new_j as u8)) {
                        Some(_exists) => {
                            match position_type {
                                SOPSCoatEnvironment::FRONT => {
                                    front_cnt += 1;
                                    front_cnct_cnt += 1;
                                }
                                SOPSCoatEnvironment::MID => {
                                    mid_cnt += 1;
                                    back_cnt -= 1;
                                    mid_cnct_cnt += 1;
                                    back_cnct_cnt -= 1;
                                }
                                _ => todo!()
                            }
                        }
                        None => { 
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
                        },
                    }
                }
            }
        }
        // print!("\t");
        let back_idx: u8 = if back_obj_seen { 10 } else { self.get_dim_idx(back_cnt, back_cnct_cnt, 3) };
        let mid_idx: u8 = if mid_obj_seen { 6 } else { self.get_dim_idx(mid_cnt, mid_cnct_cnt, 2) };
        let front_idx: u8 = if front_obj_seen { 10 } else { self.get_dim_idx(front_cnt, front_cnct_cnt, 3) };
        // println!("N:{}/{}/{}\tNs:{}/{}/{}",back_cnt, mid_cnt, front_cnt, back_cnct_cnt, mid_cnct_cnt, front_cnct_cnt);
        // TODO: Remove this hardcoding of the values. Should come from genome's dimenions
        (back_idx.clamp(0, 10), mid_idx.clamp(0, 6), front_idx.clamp(0, 10))
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
            if self.grid[new_i][new_j] == SOPSCoatEnvironment::EMPTY || self.grid[new_i][new_j] == SOPSCoatEnvironment::INCOAT || self.grid[new_i][new_j] == SOPSCoatEnvironment::COAT {
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
     * Uses BFS from the center of the object to find all connected particles
     */

    fn update_neighborhood(&mut self) {
        let mut visited: HashMap<[usize; 2], bool> = HashMap::new();
        let mut buffer: VecDeque<[usize; 2]> = VecDeque::new();
        let mut connected: HashMap<(u8, u8), bool> = HashMap::new();

        buffer.push_back(self.obj_cen_loc);

        while let Some(curr_loc) = buffer.pop_front() {
            for idx in 0..6 {
                let new_i = (curr_loc[0] as i32 + SOPSCoatEnvironment::directions()[idx].0) as usize;
                let new_j = (curr_loc[1] as i32 + SOPSCoatEnvironment::directions()[idx].1) as usize;
                if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
                    match visited.get(&[new_i, new_j]) {
                        Some(_exists) => {}
                        None => {
                            if self.grid[new_i][new_j] == SOPSCoatEnvironment::OBJECT {
                                buffer.push_back([new_i, new_j]);
                                visited.insert([new_i, new_j], true);
                            }
                            if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE
                                || (self.grid[new_i][new_j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE
                                || (self.grid[new_i][new_j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE {
                                    buffer.push_back([new_i, new_j]);
                                    visited.insert([new_i, new_j], true);
                                    connected.insert((new_i as u8, new_j as u8), true);
                                }
                        },
                    }
                }
            }
        }
        // for (key, _value) in &connected {
        //     println!("{:?} ", key);
        // }
        self.connected_par_loc = connected;
    }

    /*
     * Func to move 'n' particles in random directions in the SOPS grid
     */
    fn move_particles(&mut self, cnt: usize) {
        // let mut par_moves: Vec<(usize, (i32, i32))> = Vec::new();

        //  for _ in 0..cnt {
            // Choose a random particle for movement
            // let par_idx = SOPSCoatEnvironment::rng().sample(&self.unfrm_par());
            let par_idx = SOPSCoatEnvironment::move_frng().usize(..self.participants.len());
            // Choose a direction at random (out of the 6)
            // let move_dir = SOPSCoatEnvironment::directions()
            //             [SOPSCoatEnvironment::rng().sample(&SOPSCoatEnvironment::unfrm_dir())];
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
                    self.move_particle_to(par_idx, move_dir, front_cnt, mid_cnt);
                    self.update_neighborhood();
                }
            }

    }

    /*
     * Evaluate the measure of resultant configuration of SOPS grid
     * #. of total edges between every particle + #. of edges between particles of same color
     */
    pub fn evaluate_fitness(&self) -> f32 {
        let mut empty_inner_coat = 0;
        let mut empty_outer_coat = 0;
        // let mut free_uncnct = 0;
        // let mut free_cnct = 0;
        // let mut total_par = 0;
        for i in 0..self.size {
            for j in 0..self.size {
                // match self.grid[i][j] {
                //     1 | 2 | 5 | 6 | 8 | 9 => total_par += 1,
                //     _ => {}
                // }
                match self.grid[i][j] {
                    SOPSCoatEnvironment::INCOAT => empty_inner_coat += 1,
                    SOPSCoatEnvironment::COAT => empty_outer_coat += 1,
                    // SOPSCoatEnvironment::PARTICLE => free_uncnct += 1,
                    // SOPSCoatEnvironment::CNCTPARTICLE => free_cnct += 1,
                    _ => {}
                }
            }
        }
        let c1 =  ((self.max_inner - empty_inner_coat) as f32) / (self.max_inner as f32);
        let c2 = ((self.max_outer - empty_outer_coat) as f32) / (self.max_outer as f32);

        // println!("Free Particles: {}", (free_uncnct + free_cnct));
        // println!("Expected Free Particles: {}", ((empty_inner_coat) + (empty_outer_coat)));

        // println!("Particles: {}", total_par);
        // println!("Expected Particles: {}", self.participants.len());

        self.w1 * c1 + self.w2 * c2
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
        // self.participants.iter().for_each(|p| clr_particles[(p.state-1) as usize] +=1);
        clr_particles
    }
}
