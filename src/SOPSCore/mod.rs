pub mod separation;
pub mod coating;
pub mod locomotion;
use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};
use std::usize;
use std::collections::HashMap;

/*
 * A particle in SOPS grid used in all the behavior's SOPS grids
 *  */
struct Particle {
    x: u8,
    y: u8,
    state: u8
}

/*
 * Main Class for the Aggregation Behaviour Experiments on SOPS grid
 * Defines how the genome is interpreted and how each transition of
 * particles is derived from the the genome. Also provides final SOPS
 * grid evaluations to assess the fitness score of the genome
 * NOTE: Extend/Refer this to create new Behaviour classes
 *  */
pub struct SOPSEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    phenotype: [[[u8; 4]; 3]; 4],
    sim_duration: u64,
    fitness_val: f64,
    size: usize,
    max_fitness: u64,
    arena_layers: u16,
    particle_layers: u16,
    granularity: u8
}


impl SOPSEnvironment {

    const EMPTY: u8 = 0;
    const PARTICLE: u8 = 1;
    const BOUNDARY: u8 = 2;

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
    fn gene_probability() -> Vec<u16> {
        vec![1000, 500, 250, 125, 63, 31, 16, 8, 4, 2, 1]
    }

    // granularity is 6
    #[inline]
    fn theory_gene_probability() -> Vec<u16> {
        vec![1000, 167, 28, 5, 1, 1]
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        Uniform::new(0, self.participants.len())
    }

    /*
     * Initialize a SOPS grid and place particles based on particle layer and arena layer count
     * Parameters Particle layers and Arena layers refer to the complete hexagonal lattice layers
     * of the SOPS grid and this also defines the total density of particles in the arena.
     * Calculates Max edge count possible for all the particles irrespective of the state
     * NOTE: Use the Same random Seed value to get the same random init config
     *  */
    pub fn init_sops_env(genome: &[[[u8; 4]; 3]; 4], arena_layers: u16, particle_layers: u16, seed: u64, granularity: u8) -> Self {
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
                grid[i as usize][(i+arena_layers+j) as usize] = SOPSEnvironment::BOUNDARY;
                grid[(i+arena_layers+j) as usize][i as usize] = SOPSEnvironment::BOUNDARY;
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
                    state: 0
                });
                grid[i][j] = SOPSEnvironment::PARTICLE;
            }
            
        }

        SOPSEnvironment {
            grid,
            participants,
            phenotype: *genome,
            sim_duration: (num_particles as u64).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_fitness: agg_edge_cnt,
            arena_layers,
            particle_layers,
            granularity
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
     * Func to calculate a particle's neighbor count
     *  */
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

    /*
     * Func to calculate a particle's extended neighbor count
     *  */
    fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8) {
        let mut back_cnt = 0;
        let mut mid_cnt = 0;
        let mut front_cnt = 0;
        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();
        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == move_i) & (new_j == move_j)) {
                seen_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] == SOPSEnvironment::PARTICLE {
                    back_cnt += 1;
                }
            }
        }
        // Neighborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) & !((new_i == particle.x.into()) & (new_j == particle.y.into())) {
                let mut position_type = SOPSEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSEnvironment::MID;
                    }
                    None => {},
                }
                if self.grid[new_i][new_j] == SOPSEnvironment::PARTICLE {
                    match position_type {
                        SOPSEnvironment::FRONT => {
                            front_cnt += 1;
                        }
                        SOPSEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                        }
                        _ => todo!()
                    }
                }
            }
        }
        // TODO: Remove this hardcoding of the values. Should come from genome's dimenions
        (back_cnt.clamp(0, 3), mid_cnt.clamp(0, 2), front_cnt.clamp(0, 3))
    }

    /*
     * Func to check if the proposed move is possible or not for a particle
     *  */
     fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // Move particle if movement is within bound
        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            // check to see if move is valid ie. within bounds and not in an already occupied location
            if self.grid[new_i][new_j] == SOPSEnvironment::PARTICLE || self.grid[new_i][new_j] == SOPSEnvironment::BOUNDARY {
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
     * Func to make changes to SOPS grid by moving a particle in a given direction <- most basic operation
     *  */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let mut particle = &mut self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;
        // move the particle
        self.grid[particle.x as usize][particle.y as usize] = SOPSEnvironment::EMPTY;
        self.grid[new_i][new_j] = SOPSEnvironment::PARTICLE;
        particle.x = new_i as u8;
        particle.y = new_j as u8;
        return true;
    }

    /*
     * Func to move 'n' particles in random directions in the SOPS grid
     *  */
    fn move_particles(&mut self, cnt: usize) {
        
        // Choose a random particle for movement
        let par_idx = SOPSEnvironment::move_frng().usize(..self.participants.len());

        // Choose a random direction and validate
        let move_dir = SOPSEnvironment::directions()
                            [SOPSEnvironment::move_frng().usize(..SOPSEnvironment::directions().len())];
            
        if self.particle_move_possible(par_idx, move_dir) {
            // Get the neighborhood configuration
            let (back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);
            // Move basis probability given by the genome for moving for given configuration
            let move_prb = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize];
            if SOPSEnvironment::move_frng().u16(1_u16..=1000) <= SOPSEnvironment::gene_probability()[move_prb as usize]
            {
                self.move_particle_to(par_idx, move_dir);
            }
        }
    }

    /*
     * Evaluate the measure of resultant configuration of SOPS grid
     * #. of total edges between every particle
     */
    pub fn evaluate_fitness(&self) -> u32 {
        let edges = self.participants.iter().fold(0, |sum: u32, particle| {
            sum + self.get_neighbors_cnt(particle.x, particle.y) as u32
        });
        edges / 2
    }

    /*
     * Move a single particle at a time for 'sim_duration = fn(#. of particles)' times
     *  */
    pub fn simulate(&mut self, take_snaps: bool) -> u32 {
        for step in 0..self.sim_duration {
            self.move_particles(1 as usize);
            if take_snaps && (step == (self.participants.len() as u64) || step == (self.participants.len() as u64).pow(2)) {
                self.print_grid();
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

    pub fn get_participant_cnt(&self) -> usize {
        self.participants.len()
    }
}
