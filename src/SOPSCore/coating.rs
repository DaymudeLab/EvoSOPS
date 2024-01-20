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
    max_inner: u16,
    max_outer: u16,
    lookup_dim_idx: HashMap<(u8, u8, u8), u8>,
    // obj_cen_loc: [usize; 2],
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
     // handcrafted
     /*
    pub fn init_sops_env(genome: &[[[u8; 10]; 6]; 10], arena_layers: u16, object_layers: u16, coat_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32) -> Self {
        let grid_size = (arena_layers*2 + 1) as usize;
        let mut grid = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];

        let coat_locs: [(u8, u8); 72] = [(11, 8), (11, 9), (11, 10), (11, 11), (12, 11), (12, 12), (12, 13), (13, 11), (13, 12), (13, 13), (13, 14), (13, 15), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (15, 13), (15, 14), (15, 15), (15, 16), (15, 17), (15, 18), (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (17, 11), (17, 12), (17, 13), (18, 8), (18, 9), (18, 10), (18, 11), (25, 22), (25, 23), (25, 24), (25, 25), (26, 25), (26, 26), (26, 27), (27, 25), (27, 26), (27, 27), (27, 28), (27, 29), (28, 27), (28, 28), (28, 29), (28, 30), (28, 31), (28, 32), (29, 27), (29, 28), (29, 29), (29, 30), (29, 31), (29, 32), (30, 25), (30, 26), (30, 27), (30, 28), (30, 29), (31, 25), (31, 26), (31, 27), (32, 22), (32, 23), (32, 24), (32, 25)];

        let inner_locs: [(u8, u8); 90] = [(10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (11, 7), (11, 12), (11, 13), (12, 8), (12, 9), (12, 10), (12, 14), (12, 15), (13, 10), (13, 16), (13, 17), (13, 18), (14, 11), (14, 12), (14, 19), (15, 10), (15, 11), (15, 12), (15, 19), (16, 10), (16, 16), (16, 17), (16, 18), (16, 19), (17, 7), (17, 8), (17, 9), (17, 10), (17, 14), (17, 15), (17, 16), (18, 7), (18, 12), (18, 13), (18, 14), (19, 8), (19, 9), (19, 10), (19, 11), (19, 12), (24, 21), (24, 22), (24, 23), (24, 24), (24, 25), (25, 21), (25, 26), (25, 27), (26, 22), (26, 23), (26, 24), (26, 28), (26, 29), (27, 24), (27, 30), (27, 31), (27, 32), (28, 25), (28, 26), (28, 33), (29, 24), (29, 25), (29, 26), (29, 33), (30, 24), (30, 30), (30, 31), (30, 32), (30, 33), (31, 21), (31, 22), (31, 23), (31, 24), (31, 28), (31, 29), (31, 30), (32, 21), (32, 26), (32, 27), (32, 28), (33, 22), (33, 23), (33, 24), (33, 25), (33, 26)];

        let outer_locs: [(u8, u8); 94] = [(9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (10, 6), (10, 12), (10, 13), (11, 6), (11, 14), (11, 15), (12, 7), (12, 16), (12, 17), (12, 18), (13, 8), (13, 9), (13, 19), (14, 10), (14, 20), (15, 9), (15, 20), (16, 6), (16, 7), (16, 8), (16, 9), (16, 20), (17, 6), (17, 17), (17, 18), (17, 19), (17, 20), (18, 6), (18, 15), (18, 16), (18, 17), (19, 7), (19, 13), (19, 14), (19, 15), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12), (20, 13), (23, 20), (23, 21), (23, 22), (23, 23), (23, 24), (23, 25), (24, 20), (24, 26), (24, 27), (25, 20), (25, 28), (25, 29), (26, 21), (26, 30), (26, 31), (26, 32), (27, 22), (27, 23), (27, 33), (28, 24), (28, 34), (29, 23), (29, 34), (30, 20), (30, 21), (30, 22), (30, 23), (30, 34), (31, 20), (31, 31), (31, 32), (31, 33), (31, 34), (32, 20), (32, 29), (32, 30), (32, 31), (33, 21), (33, 27), (33, 28), (33, 29), (34, 22), (34, 23), (34, 24), (34, 25), (34, 26), (34, 27)];

        
        let num_particles = inner_locs.len() as u16 + outer_locs.len() as u16;

        let num_inner_pos = inner_locs.len() as u16;
        let num_outer_pos = num_particles - num_inner_pos;

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
                if coat_locs.iter().any(|&x| x.0 == (i as u8) && x.1== (j as u8)) {
                    grid[i][j] = SOPSCoatEnvironment::OBJECT;
                }
                else if inner_locs.iter().any(|&x| x.0 == (i as u8) && x.1== (j as u8)) {
                    grid[i][j] = SOPSCoatEnvironment::INCOAT;
                }
                else if outer_locs.iter().any(|&x| x.0 == (i as u8) && x.1== (j as u8)) {
                    grid[i][j] = SOPSCoatEnvironment::COAT;
                }
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
            max_inner: num_inner_pos,
            max_outer: num_outer_pos,
            granularity,
            lookup_dim_idx,
            w1,
            w2,
            // obj_cen_loc: [(obj_loc[0]+total_coat as usize), (obj_loc[1]+total_coat as usize)],
        }
    }
     */

    pub fn init_sops_env(genome: &[[[u8; 10]; 6]; 10], arena_layers: u16, object_layers: u16, coat_layers: u16, seed: u64, granularity: u8, w1: f32, w2: f32) -> Self {
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
            max_inner: num_inner_pos,
            max_outer: num_outer_pos,
            granularity,
            lookup_dim_idx,
            w1,
            w2,
            // obj_cen_loc: [(obj_loc[0]+total_coat as usize), (obj_loc[1]+total_coat as usize)],
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
                    // match self.connected_par_loc.get(&(i as u8, j as u8)) {
                    //     Some(_exists) => { print!(" 9 "); }
                    //     None => { print!(" 1 "); },
                    // }
                    print!(" 1 ");
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
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE {
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
                else if self.grid[new_i][new_j] == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::INCOAT) == SOPSCoatEnvironment::PARTICLE || (self.grid[new_i][new_j] - SOPSCoatEnvironment::COAT) == SOPSCoatEnvironment::PARTICLE {
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
        // println!("N:{}/{}/{}\tNs:{}/{}/{}",back_cnt, mid_cnt, front_cnt, back_cnct_cnt, mid_cnct_cnt, front_cnct_cnt);
        // TODO: Remove this hardcoding of the values. Should come from genome's dimenions
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
                    self.move_particle_to(par_idx, move_dir);
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
