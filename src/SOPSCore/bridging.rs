use super::{Particle, SOPSEnvironment};
use priority_queue::PriorityQueue;
use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::collections::VecDeque;

/*
 * Main Class for the Bridging Behaviour Expirament on SOPS grid.
 * Defines how the genome is interpreted and how much each transaction of
 * particles is derived from the genome. Also provides final SOPS grid evaluations
 * to assess the fitness score of the genome.
 */

pub struct SOPSBridEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    anchors: Vec<Particle>,
    phenotype: [[[u8; 10]; 6]; 10],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    max_fitness: f32,
    arena_layers: u16,
    particle_layers: u16,
    granularity: u8,
    lookup_dim_idx: HashMap<(u8, u8, u8), u8>,
    participant_start_count: u64,
}

impl SOPSBridEnvironment {
    //Defines grid space
    const EMPTY_LAND: u8 = 0;
    const PARTICLE_LAND: u8 = 1;
    const EMPTY_OFFLAND: u8 = 2;
    const PARTICLE_OFFLAND: u8 = 3;
    const ANCHOR: u8 = 4;
    const BOUNDARY: u8 = 5;

    //Defines neighborhood section
    const BACK: u8 = 0;
    const MID: u8 = 1;
    const FRONT: u8 = 2;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        return rand::thread_rng();
    }

    #[inline]
    fn seed_rng(seed: u64) -> rngs::StdRng {
        return rand::rngs::StdRng::seed_from_u64(seed);
    }

    #[inline]
    fn move_frng() -> fastrand::Rng {
        return fastrand::Rng::new();
    }

    #[inline]
    fn grid_rng(size_s: usize, size_e: usize) -> Uniform<usize> {
        return Uniform::new(size_s, size_e);
    }

    #[inline]
    fn unfrm_move() -> Uniform<u64> {
        return Uniform::<u64>::new(0, 1000);
    }

    #[inline]
    fn unfrm_dir() -> Uniform<usize> {
        return Uniform::new(0, 6);
    }

    #[inline]
    fn directions() -> Vec<(i32, i32)> {
        return vec![(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)];
    }

    // granularity is 10
    #[inline]
    fn gene_probability() -> Vec<u16> {
        vec![1000, 500, 250, 125, 63, 31, 16, 8, 4, 2, 1]
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        return Uniform::new(0, self.participants.len());
    }

    /**
     * Description:
     *  Initialize a SOPS grid and place particles based on particle layer and arena layer count
     *  Parameters Particle layers and Arena layers refer to the complete hexagonal lattice layers
     *  of the SOPS grid and this also defines the total density of particles in the arena.
     *  Calculates Max edge count possible for all the particles irrespective of the color
     *  Calculates Max edge count possible for all the particles of the same color
     * NOTE: Use the Same random Seed value to get the same random init config
     * Parameters:
     *  - genome: TODO
     *  - arena_layers: TODO
     *  - particle_layers: TODO
     *  - seed: TODO
     *  - granularity: TODO
     * Return:
     *  TODO
     */
    pub fn init_sops_env(
        genome: &[[[u8; 10]; 6]; 10],
        arena_layers: u16,
        particle_layers: u16,
        seed: u64,
        granularity: u8,
    ) -> Self {
        let grid_size = (arena_layers * 2 + 1) as usize;
        let mut grid: Vec<Vec<u8>> = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let mut anchors: Vec<Particle> = vec![];
        let k = 3 * particle_layers;
        let mut grid_rng = SOPSBridEnvironment::seed_rng(seed);

        //Places EMPTY_OFFLAND in a pyramid
        let square_width = 17;
        for i in 0..grid_size {
            for j in 0..grid_size {
                if j >= grid_size - square_width && i >= grid_size - square_width {
                    grid[i as usize][j as usize] = SOPSBridEnvironment::EMPTY_OFFLAND;
                }
            }
        }

        //init grid bounds
        for i in 0..arena_layers {
            let mut j = 1;
            while i + arena_layers + j < (grid_size as u16) {
                grid[i as usize][(i + arena_layers + j) as usize] = SOPSBridEnvironment::BOUNDARY;
                grid[(i + arena_layers + j) as usize][i as usize] = SOPSBridEnvironment::BOUNDARY;
                j += 1;
            }
        }

        let anchor_coords: [(u8, u8); 2] = [(9, 22), (22, 9)];
        for coordinate in anchor_coords {
            grid[coordinate.0 as usize][coordinate.1 as usize] = SOPSBridEnvironment::ANCHOR;
            anchors.push(Particle {
                x: coordinate.0 as u8,
                y: coordinate.1 as u8,
                color: 0,
            });
        }

        //Removes offland points for column 1,2 and row 1,2
        for i in 0..2 {
            for j in 0..grid_size {
                if grid[i as usize][j as usize] == SOPSBridEnvironment::EMPTY_OFFLAND {
                    grid[i as usize][j as usize] = SOPSBridEnvironment::EMPTY_LAND;
                }

                if grid[j as usize][i as usize] == SOPSBridEnvironment::EMPTY_OFFLAND {
                    grid[j as usize][i as usize] = SOPSBridEnvironment::EMPTY_LAND;
                }
            }
        }

        let particle_coord: [(u8, u8); 52] = [
            (8, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (8, 12),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (9, 8),
            (9, 9),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 21),
            (10, 8),
            (10, 9),
            (11, 8),
            (11, 9),
            (12, 8),
            (12, 9),
            (13, 8),
            (13, 9),
            (14, 8),
            (14, 9),
            (15, 8),
            (15, 9),
            (16, 8),
            (16, 9),
            (17, 8),
            (17, 9),
            (18, 8),
            (18, 9),
            (19, 8),
            (19, 9),
            (20, 8),
            (20, 9),
            (21, 8),
            (21, 9),
        ];

        for coord in particle_coord {
            grid[coord.0 as usize][coord.1 as usize] = SOPSBridEnvironment::PARTICLE_LAND;

            participants.push(Particle {
                x: coord.0,
                y: coord.1,

                //Color = 0 when Particle is on land.
                //Color = 1 when Particle is offland.
                color: 0,
            });
        }

        let lookup_dim_idx: HashMap<(u8, u8, u8), u8> = ([
            ((0, 0, 2), 0), // (0,0,4)
            ((1, 0, 2), 1),
            ((1, 1, 2), 2),
            ((2, 0, 2), 3),
            ((2, 1, 2), 4),
            ((2, 2, 2), 5),
            ((0, 0, 3), 0), // (0,0,6)
            ((1, 0, 3), 1), //
            ((1, 1, 3), 2), //
            ((2, 0, 3), 3), //
            ((2, 1, 3), 4), //
            ((2, 2, 3), 5), //
            ((3, 0, 3), 6), //
            ((3, 1, 3), 7), //
            ((3, 2, 3), 8), //
            ((3, 3, 3), 9), //
        ])
        .into();

        let num_participants = participants.len() as u64;

        SOPSBridEnvironment {
            grid,
            participants,
            anchors,
            phenotype: *genome,
            sim_duration: (num_participants).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            max_fitness: 1.0,
            arena_layers,
            particle_layers,
            granularity,
            lookup_dim_idx,
            participant_start_count: num_participants,
        }
    }

    pub fn print_grid(&self) {
        println!("SOPS grid");
        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                print!(" {} ", self.grid[i][j])
            }
            println!("")
        }
    }

    /*
     * Func to get index into a genome's dimension
     */
    fn get_dim_idx(&self, all_cnt: u8, offland_cnt: u8, all_possible_cnt: u8) -> u8 {
        match self
            .lookup_dim_idx
            .get(&(all_cnt, offland_cnt, all_possible_cnt))
        {
            Some(idx) => {
                return *idx;
            }
            None => 0,
        }
    }

    /**
     * Description:
     *  Calculates the amount of neighbor in the particles extended neighborhood.
     * Parameters:
     *  particle_idx: Index of particle in self.partipants
     *  direction: (i32, i32) tuple representing the direciton of the particle.
     * Return:
     *  A (u8, u8, u8) tuple representing the amount of neighbors in the back, middle, and front.
     */
    fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8) {
        let mut back_cnt = 0;
        let mut offland_back_cnt = 0;
        let mut mid_cnt = 0;
        let mut offland_mid_cnt = 0;
        let mut front_cnt = 0;
        let mut offland_front_cnt = 0;

        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();

        // Neighborhood for original position
        for idx in 0..6 {
            let new_i = (particle.x as i32 + SOPSBridEnvironment::directions()[idx].0) as usize;
            let new_j = (particle.y as i32 + SOPSBridEnvironment::directions()[idx].1) as usize;
            if (0..self.grid.len()).contains(&new_i)
                & (0..self.grid.len()).contains(&new_j)
                & !((new_i == move_i) & (new_j == move_j))
            {
                seen_neighbor_cache.insert([new_i, new_j], true);
                if self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_LAND
                    || self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_OFFLAND
                    || self.grid[new_i][new_j] == SOPSBridEnvironment::ANCHOR
                {
                    back_cnt += 1;
                }

                if self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_OFFLAND {
                    offland_back_cnt += 1;
                }
            }
        }

        //Nieghborhood for new position
        for idx in 0..6 {
            let new_i = (move_i as i32 + SOPSBridEnvironment::directions()[idx].0) as usize;
            let new_j = (move_j as i32 + SOPSBridEnvironment::directions()[idx].1) as usize;

            if (0..self.grid.len()).contains(&new_i)
                & (0..self.grid.len()).contains(&new_j)
                & !((new_i == particle.x.into()) & (new_j == particle.y.into()))
            {
                let mut position_type = SOPSBridEnvironment::FRONT;
                match seen_neighbor_cache.get(&[new_i, new_j]) {
                    Some(_exists) => {
                        position_type = SOPSBridEnvironment::MID;
                    }
                    None => {}
                }

                if self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_LAND
                    || self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_OFFLAND
                    || self.grid[new_i][new_j] == SOPSBridEnvironment::ANCHOR
                {
                    match position_type {
                        SOPSBridEnvironment::FRONT => {
                            front_cnt += 1;

                            if self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_OFFLAND {
                                offland_front_cnt += 1;
                            }
                        }
                        SOPSBridEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;

                            if self.grid[new_i][new_j] == SOPSBridEnvironment::PARTICLE_OFFLAND {
                                offland_back_cnt -= 1;
                                offland_mid_cnt += 1;
                            }
                        }
                        _ => todo!(),
                    }
                }
            }
        }

        let back_idx: u8 = self.get_dim_idx(back_cnt, offland_back_cnt, 3);
        let mid_idx: u8 = self.get_dim_idx(mid_cnt, offland_mid_cnt, 2);
        let front_idx: u8 = self.get_dim_idx(front_cnt, offland_front_cnt, 3);

        return (
            back_idx.clamp(0, 15),
            mid_idx.clamp(0, 8),
            front_idx.clamp(0, 15),
        );
    }

    /**
     * Description:
     *  Determines if move is possible.
     * Parameters:
     *  particle_idx: Index of particle in self.particpants.
     *  direction: (i32, i32) tuple of the direction of move.
     * Return:
     *  A boolean value representing if the move is possible.
     */
    fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &self.participants[particle_idx];
        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;

        if (0..self.grid.len()).contains(&new_i) & (0..self.grid.len()).contains(&new_j) {
            if self.grid[new_i][new_j] == SOPSBridEnvironment::EMPTY_LAND
                || self.grid[new_i][new_j] == SOPSBridEnvironment::EMPTY_OFFLAND
            {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    /**
     * Description:
     *  Changes the particle on the SOPS grid by moving a particle in a given direction
     * Parameters:
     *  particle_idx: The index of the particle in self.participants.
     *  direction: (i32, i32) tuple representing the direciton of the move.
     * Return:
     *  A boolean value representing whether the move was made.
     */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        let particle = &mut self.participants[particle_idx];

        let new_i = (particle.x as i32 + direction.0) as usize;
        let new_j = (particle.y as i32 + direction.1) as usize;

        if self.grid[particle.x as usize][particle.y as usize] == SOPSBridEnvironment::PARTICLE_LAND
        {
            self.grid[particle.x as usize][particle.y as usize] = SOPSBridEnvironment::EMPTY_LAND;
        } else {
            self.grid[particle.x as usize][particle.y as usize] =
                SOPSBridEnvironment::EMPTY_OFFLAND;
        }

        if self.grid[new_i][new_j] == SOPSBridEnvironment::EMPTY_LAND {
            self.grid[new_i][new_j] = SOPSBridEnvironment::PARTICLE_LAND;
            particle.color = 0;
        } else {
            self.grid[new_i][new_j] = SOPSBridEnvironment::PARTICLE_OFFLAND;
            particle.color = 1;
        }

        particle.x = new_i as u8;
        particle.y = new_j as u8;

        return true;
    }

    /**
     * Description:
     *  Move 'n' particles in random directions in the SOPS grid.
     * Parameters:
     *  cnt: A usize value representing the amount of particles to randomly move.
     */
    fn move_particles(&mut self, cnt: usize) {
        //See SOPSCORE::mod.rs for parallel execution commentry.
        // for _ in 0..cnt {
        let par_idx = SOPSBridEnvironment::move_frng().usize(..self.participants.len());
        let move_dir = SOPSBridEnvironment::directions()
            [SOPSBridEnvironment::move_frng().usize(..SOPSBridEnvironment::directions().len())];

        if self.particle_move_possible(par_idx, move_dir) {
            let (back_cnt, mid_cnt, front_cnt) = self.get_ext_neighbors_cnt(par_idx, move_dir);

            let move_prb = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize];
            if SOPSBridEnvironment::move_frng().u16(1_u16..=1000)
                <= SOPSBridEnvironment::gene_probability()[move_prb as usize]
            {
                self.move_particle_to(par_idx, move_dir);
            }
        }
        // }
    }

    /**
     * Description:
     *  Finds the unit vector from the current_point to anchor. Uses round to determine which direction on the grid is closest.
     * Return:
     *  A (i8, i8) tuple representing the unit vector (rounded) in the direction of the anchor point.
     */
    fn direction_to(&self, current_point: &(u8, u8), anchor: &(u8, u8)) -> (i8, i8) {
        let vector: (i8, i8) = (
            anchor.0 as i8 - current_point.0 as i8,
            anchor.1 as i8 - current_point.1 as i8,
        );
        let vector_magnitude: f32 =
            (((vector.0 as i32).pow(2) + (vector.1 as i32).pow(2)) as f32).sqrt();

        return (
            (vector.0 as f32 / vector_magnitude).round() as i8,
            (vector.1 as f32 / vector_magnitude).round() as i8,
        );
    }

    /**
     * Description:
     *  Checks bridge distance from anchor 1 to anchor 2.
     * Return:
     *  An u32 value representing the amount of particles between anchor1 and anchor2.
     */
    fn bridge_distance(&self) -> u32 {
        assert!(
            self.anchors.len() == 2,
            "Distance function is made for 2 anchors!"
        );

        let mut min_distance: u32 = u32::MAX;
        if let Some(distance_matrix) = self.get_distance_matrix() {
            for i in 0..=2 {
                let checking_x: i8 = self.anchors[1].x as i8 + i - 1;
                if self.in_bounds(checking_x as i32) {
                    for j in 0..=2 {
                        let checking_y: i8 = self.anchors[1].y as i8 + j - 1;
                        if self.in_bounds(checking_y as i32) {
                            let neighbor_grid = self.grid[checking_x as usize][checking_y as usize];
                            if neighbor_grid == SOPSBridEnvironment::PARTICLE_LAND
                                || neighbor_grid == SOPSBridEnvironment::PARTICLE_OFFLAND
                            {
                                let checking_distance =
                                    distance_matrix[checking_x as usize][checking_y as usize];
                                if (checking_distance as u32) < min_distance {
                                    min_distance = checking_distance as u32 + 1;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            return 0;
        }

        if min_distance == u32::MAX {
            self.print_grid();
            panic!("Did not find true min distance :(");
        }

        //Accounts for anchor2
        return min_distance + 1;
    }

    /**
     * Description:
     *  Caluclates optimal distance from anchor1 to anchor2.
     * Return:
     *  A u32 value representing the amount of particles between.
     */
    fn bridge_optimal_distance(&self) -> u32 {
        assert!(
            self.anchors.len() == 2,
            "Optimal distance function is made for 2 anchors!"
        );

        let anchor2_coordinates: (u8, u8) = (self.anchors[1].x, self.anchors[1].y);

        let mut current_coordinates: (u8, u8) = (self.anchors[0].x, self.anchors[0].y);
        let mut bridge_length: u32 = 0;

        while current_coordinates != anchor2_coordinates {
            bridge_length += 1;

            //Determines direction of anchor2_coordinate
            let unit_vector: (i8, i8) =
                self.direction_to(&current_coordinates, &anchor2_coordinates);
            let check_x: i8 = current_coordinates.0 as i8 + unit_vector.0;
            let check_y: i8 = current_coordinates.1 as i8 + unit_vector.1;

            if check_x >= 0 && check_y >= 0 {
                current_coordinates = (check_x as u8, check_y as u8);
            } //else {
              //     println!("Out of bounds at ({},{}) for ({},{}).", check_x, check_y, current_coordinates.0, current_coordinates.1);
              // }
        }

        //Accounts for anchor2
        return bridge_length + 1;
    }

    /**
     * Description:
     *  Determines ratio of onland to total particles. Favors more particles onland.
     * Return:
     *  A f32 value representing the ratio of onland to total particles.
     */
    fn bridge_resource(&self) -> f32 {
        let mut total_on_land_particles: u32 = 0;
        let mut total_particles: u32 = 0;

        for particle in self.participants.iter() {
            total_particles += 1;

            if particle.color == 0 {
                total_on_land_particles += 1;
            }
        }

        return total_on_land_particles as f32 / total_particles as f32;
    }

    /**
     * Description:
     *  Determines whether the value is within the grid bounds.
     * Return:
     *  A bool representing whether the value is inbounds.
     */
    fn in_bounds(&self, val: i32) -> bool {
        if val >= 0 && val < self.grid.len() as i32 {
            return true;
        }

        return false;
    }

    /**
     * Description:
     *  Determines bridge strength by finding max tension among the particle within the bridge.
     * Return:
     *  A f32 value in [0,1] representing the strength of the current bridge. The value 1 is a maximum strength bridge.
     */
    pub fn bridge_strength(&self) -> f32 {
        let mut distance_matrix: Vec<Vec<i32>> = vec![vec![-1; self.grid.len()]; self.grid.len()];

        let mut offland_particles: Vec<&Particle> = vec![];
        let mut border_particles: Vec<&Particle> = vec![];

        //Sets tension_rating for onland and determines offland particles
        for particle in self.participants.iter() {
            let x: usize = particle.x as usize;
            let y: usize = particle.y as usize;

            if self.grid[x][y] == SOPSBridEnvironment::PARTICLE_LAND {
                distance_matrix[particle.x as usize][particle.y as usize] = 0;
            } else if self.grid[x][y] == SOPSBridEnvironment::PARTICLE_OFFLAND {
                offland_particles.push(particle);

                //Checks neighborhood for a land particle
                let mut border_particle: bool = false;
                for i in 0..=2 {
                    let checking_x: i32 = x as i32 + i - 1;
                    if self.in_bounds(checking_x) {
                        for j in 0..=2 {
                            let checking_y: i32 = y as i32 + j - 1;
                            if self.in_bounds(checking_y) {
                                if self.grid[checking_x as usize][checking_y as usize]
                                    == SOPSBridEnvironment::PARTICLE_LAND
                                {
                                    border_particle = true;
                                }
                            }
                        }
                    }
                }

                if border_particle {
                    border_particles.push(particle);
                }
            }
        }

        let gravity_constant: u8 = 9;
        let mut max_tension: u32 = 0;

        let mut queue: PriorityQueue<(u8, u8), Reverse<i32>> = PriorityQueue::new();
        let mut closed: Vec<(u8, u8)> = vec![];

        for particle in border_particles {
            queue.push((particle.x, particle.y), Reverse(0));
        }

        while !queue.is_empty() {
            let particle = queue.pop().expect("Empty priority queue.");
            let x = particle.0 .0;
            let y = particle.0 .1;

            distance_matrix[x as usize][y as usize] = particle.1 .0 + 1;
            closed.push((x, y));

            //Get particle neighbors
            for i in 0..=2 {
                let checking_x: i32 = x as i32 + i - 1;
                if self.in_bounds(checking_x) {
                    for j in 0..=2 {
                        let checking_y: i32 = y as i32 + j - 1;
                        if self.in_bounds(checking_y) {
                            if self.grid[checking_x as usize][checking_y as usize]
                                == SOPSBridEnvironment::PARTICLE_OFFLAND
                                && !closed.contains(&(checking_x as u8, checking_y as u8))
                                && !queue.clone().iter().any(|(value, _)| {
                                    value.0 == checking_x as u8 && value.1 == checking_y as u8
                                })
                            {
                                queue.push(
                                    (checking_x as u8, checking_y as u8),
                                    Reverse(particle.1 .0 + 1),
                                );
                            }
                        }
                    }
                }
            }
        }

        //Find offland_particle with max tension
        for particle in offland_particles.iter() {
            let x: usize = particle.x as usize;
            let y: usize = particle.y as usize;

            let distance: u32 = (distance_matrix[x][y]) as u32;

            let mut neighbors: u8 = 0;
            for i in 0..2 {
                let checking_x: i32 = x as i32 + i - 1;
                if checking_x >= 0 {
                    for j in 0..2 {
                        let checking_y: i32 = y as i32 + j - 1;
                        if checking_y >= 0 {
                            if self.grid[checking_x as usize][checking_y as usize]
                                == SOPSBridEnvironment::PARTICLE_LAND
                                || self.grid[checking_x as usize][checking_y as usize]
                                    == SOPSBridEnvironment::PARTICLE_OFFLAND
                            {
                                neighbors += 1;
                            }
                        }
                    }
                }
            }

            let tension: i32 = distance as i32 + gravity_constant as i32 - neighbors as i32;
            assert!(
                tension >= 0,
                "Tension is below 0: {}. Raise gravity constant from {}.",
                tension,
                gravity_constant
            );

            if tension as u32 > max_tension {
                max_tension = tension as u32;
            }
        }

        if max_tension == 0 {
            return 0.0;
        }

        let max_tension_allowed: f32 = 5.0;

        //Minimum tension = 1
        //Maps [1,20] -> [1,0]
        let min_tension_rating: f32 =
            ((-1.0 / (max_tension_allowed - 1.0)) * (max_tension as f32 - 1.0)) + 1.0;

        return min_tension_rating.max(0.0);
    }

    /**
     * Description: Returns matrix holding particle distance from anchor1.
     */
    fn get_distance_matrix(&self) -> Option<Vec<Vec<i32>>> {
        //Initializes distance_matrix to -1
        let mut distance_matrix: Vec<Vec<i32>> = vec![vec![-1; self.grid.len()]; self.grid.len()];
        let mut searched_matrix: Vec<Vec<bool>> =
            vec![vec![false; self.grid.len()]; self.grid.len()];
        let mut queue: VecDeque<(u8, u8)> = VecDeque::new();
        queue.push_back((self.anchors[0].x, self.anchors[0].y));

        let mut current_particle = queue.pop_front().expect("Empty queue!");
        searched_matrix[current_particle.0 as usize][current_particle.1 as usize] = true;

        while !(current_particle.0 == self.anchors[1].x && current_particle.1 == self.anchors[1].y)
        {
            let mut min_neighbor_distance: u32 = u32::MAX;

            //Checks neighborhood
            for i in 0..=2 {
                let checking_x: i8 = current_particle.0 as i8 + i - 1;
                if checking_x >= 0 && checking_x < self.grid.len() as i8 {
                    for j in 0..=2 {
                        let checking_y: i8 = current_particle.1 as i8 + j - 1;
                        if (checking_y >= 0 && checking_y < self.grid[0].len() as i8)
                            && !(checking_x as u8 == current_particle.0
                                && checking_y as u8 == current_particle.1)
                        {
                            let neighbor_grid = self.grid[checking_x as usize][checking_y as usize];
                            if neighbor_grid == SOPSBridEnvironment::PARTICLE_LAND
                                || neighbor_grid == SOPSBridEnvironment::PARTICLE_OFFLAND
                                || neighbor_grid == SOPSBridEnvironment::ANCHOR
                            {
                                let neighbor_distance =
                                    distance_matrix[checking_x as usize][checking_y as usize];
                                if neighbor_distance == -1 {
                                    if searched_matrix[checking_x as usize][checking_y as usize]
                                        == false
                                    {
                                        queue.push_back((checking_x as u8, checking_y as u8));
                                        searched_matrix[checking_x as usize][checking_y as usize] =
                                            true;
                                    }
                                } else if neighbor_distance >= 0 {
                                    if (neighbor_distance as u32) < min_neighbor_distance {
                                        min_neighbor_distance = neighbor_distance as u32;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if current_particle.0 == self.anchors[0].x && current_particle.1 == self.anchors[0].y {
                distance_matrix[current_particle.0 as usize][current_particle.1 as usize] = 0;
            } else {
                distance_matrix[current_particle.0 as usize][current_particle.1 as usize] =
                    min_neighbor_distance as i32 + 1;
            }

            if let Some(particle_coordinates) = queue.pop_front() {
                current_particle = particle_coordinates;
            } else {
                //If queue is empty return 0
                return None;
            }
        }

        return Some(distance_matrix);
    }

    /**
     * Description:
     *  Calculates the ratio between optimal and current bridge distance.
     */
    fn calculate_distance_ratio(&self) -> f32 {
        let current_distance = self.bridge_distance();

        if current_distance == 0 {
            return 0.0;
        }

        return self.bridge_optimal_distance() as f32 / current_distance as f32;
    }

    /**
     * Description:
     *  Calculates the amount of "dying" particles
     */
    fn connectivity_metric(&mut self) -> f32 {
        let participant_count = self.participants.len();
        let mut kill_count: u32 = 0;

        let mut index = 0;
        while index < self.participants.len() {
            if let Some(particle) = self.participants.get(index) {
                //Checks for offland particle
                if particle.color == 1 {
                    let mut neighbor_flag = false;

                    //Checks neighborhood for particle
                    for i in 0..self.grid.len() {
                        let checking_x = particle.x as i32 + i as i32 - 1;
                        if self.in_bounds(checking_x as i32) {
                            for j in 0..self.grid.len() {
                                let checking_y = particle.y as i32 + j as i32 - 1;
                                if self.in_bounds(checking_y) {
                                    let grid_value =
                                        self.grid[checking_x as usize][checking_y as usize];
                                    if grid_value == SOPSBridEnvironment::PARTICLE_LAND
                                        || grid_value == SOPSBridEnvironment::PARTICLE_OFFLAND
                                        || grid_value == SOPSBridEnvironment::ANCHOR
                                    {
                                        neighbor_flag = true;
                                    }
                                }
                            }
                        }
                    }

                    if !neighbor_flag {
                        //Kill offland particle
                        kill_count += 1;
                        self.participants.remove(index);
                        index -= 1;
                    }
                }

                index += 1;
            } else {
                panic!(
                    "Tried to index particle at {}. Participant Length: {}.",
                    index,
                    self.participants.len()
                );
            }
        }

        let total_kill_count =
            self.participant_start_count as u32 - participant_count as u32 + kill_count as u32;

        return 1.0 - (total_kill_count as f32 / self.participant_start_count as f32);
    }

    /**
     * Description:
     *  Determines fitness of current configuration based on three fitness criteria:
     *      Distance, Resources, Strength
     * Return:
     *  A f32 value between 0 and 1.
     */
    pub fn evaluate_fitness(&mut self) -> f32 {
        let strength_factor: f32 = 0.0;
        let distance_factor: f32 = 4.00;
        let connectivity_factor: f32 = 6.00;
        let resource_factor: f32 = 1.00;

        let total_factor: f32 =
            strength_factor + distance_factor + connectivity_factor + resource_factor;

        //Checks connectivity
        let connectivity_measure: f32 = self.connectivity_metric();

        let distance_ratio = self.calculate_distance_ratio();
        if distance_ratio == 0.0 {
            //Returns fitness value as if the rest do not matter
            return connectivity_measure / total_factor;
        }

        let bridge_resource: f32 = self.bridge_resource();
        let bridge_strength: f32 = self.bridge_strength();

        let return_value = f32::max(
            ((distance_factor * distance_ratio)
                + (connectivity_factor * connectivity_measure)
                + (resource_factor * bridge_resource)
                + (strength_factor * bridge_strength))
                / total_factor,
            0.0,
        );

        if return_value.is_infinite() {
            panic!("Infinite fitness!");
        }

        return return_value;
    }

    /**
     * Description:
     *  Runs the entirety of the simulation.
     * Return:
     *  Returns the fitness value of the final step.
     */
    pub fn simulate(&mut self, take_snaps: bool) -> f32 {
        for step in 0..self.sim_duration {
            self.move_particles(1 as usize);
            if take_snaps
                && (step == (self.participants.len() as u64)
                    || step == (self.participants.len() as u64).pow(2))
            {
                self.print_grid();
                println!(
                    "Fitness: {}",
                    self.evaluate_fitness() / self.get_max_fitness()
                );
            }
        }

        self.fitness_val = self.evaluate_fitness();
        return self.fitness_val;
    }

    /*
     * Description:
     *   Gets the maximum fitness value possible.
     * Return:
     *   A f32 value representing the maximum fitness value. Range: [0,1]
     */
    pub fn get_max_fitness(&self) -> f32 {
        return 1.0;
    }

    /*
     * Description:
     *   Gets the amount of participants within the expirament.
     * Return:
     *   A usize value representing the amount of participants.
     */
    pub fn get_participant_cnt(&self) -> usize {
        return self.participants.len();
    }
}
