use super::Particle;
use priority_queue::PriorityQueue;
use rand::seq::index;
use rand::{distributions, SeedableRng, thread_rng};
use rand::{distributions::Uniform, rngs, Rng};
use rand_distr::num_traits::Pow;
use rand_distr::Distribution;
use core::num;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub struct SOPSBridEnvironment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    anchors: Vec<Particle>,
    phantom_participants: Vec<Particle>,
    phenotype: [[[[[u8; 2]; 2]; 4]; 3]; 4],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    granularity: u8,
    strength_weight: f32,
    distance_weight: f32,
    connectivity_weight: f32,
    resource_weight: f32
}

impl SOPSBridEnvironment {
    const EMPTY_LAND: u8 = 0;
    const PARTICLE_LAND: u8 = 1;
    const EMPTY_OFFLAND: u8 = 2;
    const PARTICLE_OFFLAND: u8 = 3;
    const ANCHOR: u8 = 4;
    const BOUNDARY: u8 = 5;

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

    pub fn init_sops_env(
        genome: &[[[[[u8; 2]; 2]; 4]; 3]; 4],
        arena_layers: u16,
        particle_layers: u16,
        gap_diagonal_length: u16,
        gap_angle_degrees: u16,
        granularity: u8,
        strength_weight: f32,
        distance_weight: f32,
        connectivity_weight: f32,
        resource_weight: f32
    ) -> Self {
        //inits grid
        let grid_size = (arena_layers * 2 + 1) as usize;
        let mut grid: Vec<Vec<u8>> = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let mut anchors: Vec<Particle> = vec![];
        let phantom_participants: Vec<Particle> = vec![];

        //Places bounderies
        for i in 0..arena_layers {
            let mut j = 1;
            while i + arena_layers + j < (grid_size as u16) {
                grid[i as usize][(i + arena_layers + j) as usize] = SOPSBridEnvironment::BOUNDARY;
                grid[(i + arena_layers + j) as usize][i as usize] = SOPSBridEnvironment::BOUNDARY;
                j += 1;
            }
        }

        let line_m1_check = |i: f32, j: f32, offset: u16| { 
            let intersection_point: (u16, u16) = (grid_size as u16 - gap_diagonal_length as u16 - offset, grid_size as u16 - gap_diagonal_length as u16 - offset);
            if gap_angle_degrees == 90 {
                i as u16 >= intersection_point.1
            } else {
                let tan_gap_angle_over_2 = (gap_angle_degrees as f32 * (std::f32::consts::PI / 360.0)).tan();
                let m1 = (tan_gap_angle_over_2 + 1.0) / (1.0 - tan_gap_angle_over_2);
                (j as u16) < (m1 * i - m1 * intersection_point.0 as f32 + intersection_point.1 as f32) as u16
            }
        };

        let line_m2_check = |i: f32, j: f32, offset: u16| {
            let intersection_point: (u16, u16) = (grid_size as u16 - gap_diagonal_length as u16 - offset, grid_size as u16 - gap_diagonal_length as u16 - offset);

            if gap_angle_degrees == 90 {
                j as u16 >= ((0.0 * i as f32 - 0.0 * intersection_point.0 as f32 + intersection_point.1 as f32) as u16)
            } else {
                let tan_gap_angle_over_2 = (gap_angle_degrees as f32 * -1.0 * (std::f32::consts::PI / 360.0)).tan();
                let m1 = (tan_gap_angle_over_2 + 1.0) / (1.0 - tan_gap_angle_over_2);
                j as u16 >= ((m1 * i as f32 - m1 * intersection_point.0 as f32 + intersection_point.1 as f32)) as u16
            }
        };

        for i in 0..grid_size {
            for j in 0..grid_size {
                if line_m1_check(i as f32,j as f32, 0) && line_m2_check(i as f32, j as f32, 0) {
                    if grid[i][j] != SOPSBridEnvironment::BOUNDARY {
                        grid[i as usize][j as usize] = SOPSBridEnvironment::EMPTY_OFFLAND;
                        continue;
                    }
                }            

                if line_m1_check(i as f32, j as f32, particle_layers as u16) && line_m2_check(i as f32, j as f32, particle_layers as u16) {
                    if grid[i][j] != SOPSBridEnvironment::BOUNDARY {
                        grid[i as usize][j as usize] = SOPSBridEnvironment::PARTICLE_LAND;
                        participants.push(Particle{
                            x: i as u16,
                            y: j as u16,
                            onland: true,
                        });
                        continue;
                    }
                }   
            }
        }

        //Places anchor at first unoccupied position
        let mut i = 0;
        while anchors.len() < 1 {
            for j in 0..grid_size {
                if grid[grid_size - 1 - i][grid_size - 1 - j] == SOPSBridEnvironment::EMPTY_LAND {
                    grid[grid_size - 1 - i][grid_size - 1 - j] = SOPSBridEnvironment::ANCHOR;
                    anchors.push(Particle{
                        x: (grid_size - 1 - i) as u16,
                        y: (grid_size - 1 - j) as u16,
                        onland: true,
                    });
                    break;
                }
            }

            i += 1;
        }

        let mut j = 0;
        while anchors.len() < 2 {
            for i in 0..grid_size {
                if grid[grid_size - i - 1][grid_size - j - 1] == SOPSBridEnvironment::EMPTY_LAND {
                    grid[grid_size - i - 1][grid_size - j - 1] = SOPSBridEnvironment::ANCHOR;
                    anchors.push(Particle{
                        x: (grid_size - i - 1) as u16,
                        y: (grid_size - j - 1) as u16,
                        onland: true,
                    });
                    break;
                }
            }

            j += 1;
        }

        let num_participants = participants.len() as u64;

        SOPSBridEnvironment {
            grid,
            participants,
            anchors,
            phantom_participants,
            phenotype: *genome,
            sim_duration: (num_participants).pow(3),
            fitness_val: 0.0,
            size: grid_size,
            granularity,
            strength_weight,
            distance_weight,
            connectivity_weight,
            resource_weight
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

    /**
     * Description:
     *  Calculates the amount of neighbor in the particles extended neighborhood.
     * Parameters:
     *  particle_idx: Index of particle in self.partipants
     *  direction: (i32, i32) tuple representing the direciton of the particle.
     * Return:
     *  A (u8, u8, u8, u8, u8) tuple representing the amount of neighbors in the back, middle, front, anchor presence, and onland.
     */
    fn get_ext_neighbors_cnt(&self, particle_idx: usize, direction: (i32, i32)) -> (u8, u8, u8, u8, u8) {
        let mut back_cnt = 0;
        let mut mid_cnt = 0;
        let mut front_cnt = 0;
        let mut anchor_cnt = 0;

        let particle = &self.participants[particle_idx];
        let move_i = (particle.x as i32 + direction.0) as usize;
        let move_j = (particle.y as i32 + direction.1) as usize;
        let mut seen_neighbor_cache: HashMap<[usize; 2], bool> = HashMap::new();

        //Neighborhood of original position
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
                    if self.grid[new_i][new_j] == SOPSBridEnvironment::ANCHOR {
                        anchor_cnt += 1;
                    }
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

                            if self.grid[new_i][new_j] == SOPSBridEnvironment::ANCHOR {
                                anchor_cnt += 1
                            }
                        }

                        SOPSBridEnvironment::MID => {
                            mid_cnt += 1;
                            back_cnt -= 1;
                        }
                        _ => todo!(),
                    }
                }
            }
        }

        return (
            back_cnt.clamp(0, 4),
            mid_cnt.clamp(0, 3),
            front_cnt.clamp(0, 4),
            anchor_cnt.clamp(0,2),
            u8::from(particle.onland).clamp(0, 2),
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
            return self.grid[new_i][new_j] == SOPSBridEnvironment::EMPTY_LAND
                || self.grid[new_i][new_j] == SOPSBridEnvironment::EMPTY_OFFLAND;
        }
        return false;
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
            particle.onland = true;
        } else {
            self.grid[new_i][new_j] = SOPSBridEnvironment::PARTICLE_OFFLAND;
            particle.onland = false;
        }

        particle.x = new_i as u16;
        particle.y = new_j as u16;
        return true;
    }

    /**
     * Description:
     *  Move 'n' particles in random directions in the SOPS grid.
     * Parameters:
     *  cnt: A usize value representing the amount of particles to randomly move.
     */
    fn move_particles(&mut self) {
        //See SOPSCORE::mod.rs for parallel execution commentry.
        let par_idx = SOPSBridEnvironment::move_frng().usize(..self.participants.len());
        let move_dir = SOPSBridEnvironment::directions()
            [SOPSBridEnvironment::move_frng().usize(..SOPSBridEnvironment::directions().len())];

        if self.particle_move_possible(par_idx, move_dir) {
            let (back_cnt, mid_cnt, front_cnt, anchor_cnt, onland_idx) = self.get_ext_neighbors_cnt(par_idx, move_dir);

            let move_prb = self.phenotype[back_cnt as usize][mid_cnt as usize][front_cnt as usize][anchor_cnt as usize][onland_idx as usize];
            if SOPSBridEnvironment::move_frng().u16(1_u16..=1000)
                <= SOPSBridEnvironment::gene_probability()[move_prb as usize]
            {
                self.move_particle_to(par_idx, move_dir);
            }
        }
    }

    /**
     * Description:
     *  Finds the unit vector from the current_point to anchor. Uses round to determine which direction on the grid is closest.
     * Return:
     *  A (i8, i8) tuple representing the unit vector (rounded) in the direction of the anchor point.
     */
    fn direction_to(&self, point_1: &(u16, u16), point_2: &(u16, u16)) -> (i16, i16) {
        let vector: (i16, i16) = (
            point_2.0 as i16 - point_1.0 as i16,
            point_2.1 as i16 - point_1.1 as i16,
        );
        let vector_magnitude: f32 =
            (((vector.0 as i32).pow(2) + (vector.1 as i32).pow(2)) as f32).sqrt();

        return (
            (vector.0 as f32 / vector_magnitude).round() as i16,
            (vector.1 as f32 / vector_magnitude).round() as i16,
        );
    }

    /**
     * Description: Returns matrix holding particle distance from anchor1.
     */
    fn get_distance_matrix(&self) -> Option<Vec<Vec<i32>>> {
        //Initializes distance_matrix to -1
        let mut distance_matrix: Vec<Vec<i32>> = vec![vec![-1; self.grid.len()]; self.grid.len()];
        let mut searched_matrix: Vec<Vec<bool>> =
            vec![vec![false; self.grid.len()]; self.grid.len()];
        let mut queue: VecDeque<(u16, u16)> = VecDeque::new();
        queue.push_back((self.anchors[0].x, self.anchors[0].y));

        let mut current_particle = queue.pop_front().expect("Empty queue!");
        searched_matrix[current_particle.0 as usize][current_particle.1 as usize] = true;

        while !(current_particle.0 == self.anchors[1].x && current_particle.1 == self.anchors[1].y)
        {
            let mut min_neighbor_distance: u32 = u32::MAX;

            //Checks neighborhood
            for direction in SOPSBridEnvironment::directions() {
                let checking_x: i32 = current_particle.0 as i32 + direction.0;
                if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                    let checking_y: i32 = current_particle.1 as i32 + direction.1;
                    if (checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)))
                        && !(checking_x as u16 == current_particle.0 && checking_y as u16 == current_particle.1)
                    {
                        let neighbor_grid = self.grid[checking_x as usize][checking_y as usize];
                        if neighbor_grid == SOPSBridEnvironment::PARTICLE_LAND
                            || neighbor_grid == SOPSBridEnvironment::PARTICLE_OFFLAND
                            || neighbor_grid == SOPSBridEnvironment::ANCHOR
                        {
                            let neighbor_distance =
                                distance_matrix[checking_x as usize][checking_y as usize];
                            if neighbor_distance == -1 {
                                if !searched_matrix[checking_x as usize][checking_y as usize]
                                {
                                    queue.push_back((checking_x as u16, checking_y as u16));
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
            for direction in SOPSBridEnvironment::directions() {
                let checking_x: i32 = self.anchors[1].x as i32 + direction.0;
                if checking_x >= 0 && (0..self.size).contains(&(checking_x as usize)) {
                    let checking_y: i32 = self.anchors[1].y as i32 + direction.1;
                    if checking_y >= 0 && (0..self.size).contains(&(checking_y as usize)) {
                        let neighbor_grid = self.grid[checking_x as usize][checking_y as usize];
                        if neighbor_grid == SOPSBridEnvironment::PARTICLE_LAND
                            || neighbor_grid == SOPSBridEnvironment::PARTICLE_OFFLAND
                        {
                            let checking_distance = distance_matrix[checking_x as usize][checking_y as usize];

                            if (checking_distance as u32) < min_distance {
                                min_distance = checking_distance as u32 + 1;
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

        let anchor2_coordinates: (u16, u16) = (self.anchors[1].x, self.anchors[1].y);

        let mut current_coordinates: (u16, u16) = (self.anchors[0].x, self.anchors[0].y);
        let mut bridge_length: u32 = 0;

        while current_coordinates != anchor2_coordinates {
            bridge_length += 1;

            //Determines direction of anchor2_coordinate
            let unit_vector: (i16, i16) =
                self.direction_to(&current_coordinates, &anchor2_coordinates);
            let check_x: i32 = current_coordinates.0 as i32 + unit_vector.0 as i32;
            let check_y: i32 = current_coordinates.1 as i32 + unit_vector.1 as i32;

            if check_x >= 0 && check_y >= 0 {
                current_coordinates = (check_x as u16, check_y as u16);
            } //else {
              //     println!("Out of bounds at ({},{}) for ({},{}).", check_x, check_y, current_coordinates.0, current_coordinates.1);
              // }
        }

        //+1 to account for anchor2
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

            if particle.onland {
                total_on_land_particles += 1;
            }
        }

        return total_on_land_particles as f32 / total_particles as f32;
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
                for direction in SOPSBridEnvironment::directions() {
                    let checking_x: i32 = x as i32 + direction.0;
                    if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                        let checking_y: i32 = y as i32 + direction.1;
                        if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)) {
                            if self.grid[checking_x as usize][checking_y as usize]
                                == SOPSBridEnvironment::PARTICLE_LAND
                            {
                                border_particle = true;
                            }
                        }
                    }
                }

                if border_particle {
                    border_particles.push(particle);
                }
            }
        }

        let mut queue: PriorityQueue<(u16, u16), Reverse<i32>> = PriorityQueue::new();
        let mut closed: Vec<(u16, u16)> = vec![];

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
            for direction in SOPSBridEnvironment::directions() {
                let checking_x: i32 = x as i32 + direction.0;
                if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                    let checking_y: i32 = y as i32 + direction.1;
                    if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize))  {
                        if self.grid[checking_x as usize][checking_y as usize]
                            == SOPSBridEnvironment::PARTICLE_OFFLAND
                            && !closed.contains(&(checking_x as u16, checking_y as u16))
                            && !queue.clone().iter().any(|(value, _)| {
                                value.0 == checking_x as u16 && value.1 == checking_y as u16
                            })
                        {
                            queue.push(
                                (checking_x as u16, checking_y as u16),
                                Reverse(particle.1 .0 + 1),
                            );
                        }
                    }
                }
            }
        }

        //Find offland_particle with max tension
        let mut max_tension: u32 = 0;

        for particle in offland_particles.iter() {
            let x: usize = particle.x as usize;
            let y: usize = particle.y as usize;

            let distance: u32 = (distance_matrix[x][y]) as u32;

            let mut neighbors: u16 = 0;
            for direction in SOPSBridEnvironment::directions() {
                let checking_x: i32 = x as i32 + direction.0;
                if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                    let checking_y: i32 = y as i32 + direction.1;
                    if self.grid[checking_x as usize][checking_y as usize]
                                == SOPSBridEnvironment::PARTICLE_LAND
                                || self.grid[checking_x as usize][checking_y as usize]
                                    == SOPSBridEnvironment::PARTICLE_OFFLAND
                            {
                                neighbors += 1;
                            }
                }
            }

            let tension: i32 = distance as i32 - neighbors as i32;

            if tension.max(0) as u32 > max_tension {
                max_tension = tension as u32;
            }
        }

        let decay_factor: f32 = 0.1;
        let tension_rating: f32 =
            std::f32::consts::E.powf(-1.0 * decay_factor as f32 * max_tension as f32);

        return tension_rating.max(0.0);
    }

    fn find_path_particles(&self, anchor1_coordinates: (u16, u16))->HashSet<(u16, u16)> {

        let mut path_particles: HashSet<(u16, u16)> = HashSet::new();

        //Calculates current path from anchor1 using BFS
        let mut found_set: HashSet<(u16, u16)> = HashSet::new();
        let mut queue: VecDeque<(u16, u16)> = VecDeque::new();

        queue.push_back(anchor1_coordinates);
        found_set.insert(anchor1_coordinates);
        while !queue.is_empty() {
            if let Some(current_particle_coordinates) = queue.pop_front() {
                path_particles.insert(current_particle_coordinates);

                //Checks neighborhood for particles
                for direction in SOPSBridEnvironment::directions() {
                    let checking_x: i32 = current_particle_coordinates.0 as i32 + direction.0;
                    if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize))
                {
                    let checking_y = current_particle_coordinates.1 as i32 + direction.1;
                    if checking_y >= 0 && (0..self.grid.len())
                        .contains(&(checking_y as usize)) {
                        if !found_set.contains(&(checking_x as u16, checking_y as u16)) {
                            if self.grid[checking_x as usize][checking_y as usize]
                                == SOPSBridEnvironment::PARTICLE_LAND
                                || self.grid[checking_x as usize][checking_y as usize]
                                    == SOPSBridEnvironment::PARTICLE_OFFLAND
                                || self.grid[checking_x as usize][checking_y as usize]
                                    == SOPSBridEnvironment::ANCHOR
                            {
                                queue.push_back((checking_x as u16, checking_y as u16));
                                found_set.insert((checking_x as u16, checking_y as u16));
                            }
                        }
                    }
                }

                }
            } else {
                panic!("add_phantom_participants: Reached end without quitting loop!");
            }
        }

        return path_particles;
    }

    /**
     * Description: Adds phantom particles to grid to connect bridge.
     */
    fn add_phantom_participants(&mut self) -> f32 {
        let anchor1_coordinates: (u16, u16) = (self.anchors[0].x, self.anchors[0].y);
        let anchor2_coordinates: (u16, u16) = (self.anchors[1].x, self.anchors[1].y);

        //Adds phantom particles iteratively
        let mut path_particles: HashSet<(u16, u16)> = self.find_path_particles(anchor1_coordinates);

        //Condition1: while !path_particles.contains(&anchor2_coordinates)
        //Condition2: while path_particles.len() != self.participants.len() + self.anchors.len()
        while path_particles.len() != self.participants.len() + self.anchors.len() {
            //Finds closest particles with consideration to anchor2
            let mut closest_path_particle: (u16, u16) = (0, 0);
            let mut closest_non_path_particle: (u16, u16) = (0, 0);
            let mut closest_particles_distance: f32 = f32::MAX;

            for non_path_particle in self.participants.iter() {
                let current_particle_coordinates = (non_path_particle.x, non_path_particle.y);
                if !path_particles.contains(&current_particle_coordinates) {
                    for path_particle_coordinates in path_particles.iter() {
                        let particle_distance = SOPSBridEnvironment::euclidean_distance(
                            &current_particle_coordinates,
                            path_particle_coordinates,
                        );

                        if particle_distance < closest_particles_distance as f32 {
                            closest_particles_distance = particle_distance;
                            closest_path_particle =
                                (path_particle_coordinates.0, path_particle_coordinates.1);
                            closest_non_path_particle = current_particle_coordinates;
                        }
                    }
                }
            }

            
            if closest_particles_distance == f32::MAX {
                closest_non_path_particle = anchor2_coordinates;

                for path_particle_coordinates in path_particles.iter() {
                    let particle_distance = SOPSBridEnvironment::euclidean_distance(
                        &closest_non_path_particle,
                        path_particle_coordinates,
                    );

                    if particle_distance < closest_particles_distance as f32 {
                        closest_particles_distance = particle_distance;
                        closest_path_particle = (path_particle_coordinates.0, path_particle_coordinates.1);
                    }
                }
            }

            //Computes opt_matrix for two points
            let mut opt_matrix: Vec<Vec<(i8, i32)>> =
                vec![vec![(0, -1); self.grid.len()]; self.grid.len()];
            let mut distance_factor: u32 = 1;

            let point1 = closest_path_particle;
            opt_matrix[point1.0 as usize][point1.1 as usize] = (0, 0);

            let mut found_point_2: bool = false;

            while !found_point_2 {
                //Checks horizontal parts of distance-square perimeter
                for direction in SOPSBridEnvironment::directions() {
                    let checking_x: i32 = point1.0 as i32 + (direction.0 * distance_factor as i32);
                    if checking_x >= 0 && checking_x < self.grid.len() as i32 {
                        //Checks (checking_x, point1.1 + distance_factor)
                        let top_horizontal_coordinate =
                            (checking_x as u32, (point1.1 as u32 + distance_factor));
                        if top_horizontal_coordinate.1 < self.grid.len() as u32 {
                            opt_matrix[top_horizontal_coordinate.0 as usize]
                                [top_horizontal_coordinate.1 as usize] = self.phantom_metric(
                                (
                                    top_horizontal_coordinate.0 as u16,
                                    top_horizontal_coordinate.1 as u16,
                                ),
                                &opt_matrix,
                            );
                            if top_horizontal_coordinate.0 == closest_non_path_particle.0.into()
                                && top_horizontal_coordinate.1 == closest_non_path_particle.1.into()
                            {
                                found_point_2 = true;
                            }
                        }

                        //Checks (checking_x, point1.1 - distance_factor)
                        let bottom_horizontal_coordinate = (
                            checking_x as u32,
                            (point1.1 as i32),
                        );
                        if bottom_horizontal_coordinate.1 >= 0 {
                            opt_matrix[bottom_horizontal_coordinate.0 as usize]
                                [bottom_horizontal_coordinate.1 as usize] = self.phantom_metric(
                                (
                                    bottom_horizontal_coordinate.0 as u16,
                                    bottom_horizontal_coordinate.1 as u16,
                                ),
                                &opt_matrix,
                            );
                            if bottom_horizontal_coordinate.0 == closest_non_path_particle.0.into()
                                && bottom_horizontal_coordinate.1
                                    == closest_non_path_particle.1.into()
                            {
                                found_point_2 = true;
                            }
                        }
                    }
                }

                for j in 1..distance_factor * 2 {
                    let checking_y: i32 = point1.1 as i32 + j as i32 - distance_factor as i32;
                    if checking_y >= 0 && checking_y < self.grid[0].len() as i32 {
                        //Checks (closet_particles.0.0 + distance_factor, checking_y)
                        let right_vertical_coorindate =
                            (point1.0 as u32 + distance_factor, checking_y);
                        if right_vertical_coorindate.0 < self.grid[0].len() as u32 {
                            opt_matrix[right_vertical_coorindate.0 as usize]
                                [right_vertical_coorindate.1 as usize] = self.phantom_metric(
                                (
                                    right_vertical_coorindate.0 as u16,
                                    right_vertical_coorindate.1 as u16,
                                ),
                                &opt_matrix,
                            );
                            if right_vertical_coorindate.0 == closest_non_path_particle.0.into()
                                && right_vertical_coorindate.1 == closest_non_path_particle.1.into()
                            {
                                found_point_2 = true;
                            }
                        }

                        //Checks (closet_particles.0.0 - distance_factor, checking_y)
                        let left_vertical_coorindate =
                            ((point1.0 as i32 - distance_factor as i32), checking_y);
                        if left_vertical_coorindate.0 >= 0 {
                            opt_matrix[left_vertical_coorindate.0 as usize]
                                [left_vertical_coorindate.1 as usize] = self.phantom_metric(
                                (
                                    left_vertical_coorindate.0 as u16,
                                    left_vertical_coorindate.1 as u16,
                                ),
                                &opt_matrix,
                            );
                            if left_vertical_coorindate.0 == closest_non_path_particle.0.into()
                                && left_vertical_coorindate.1 == closest_non_path_particle.1.into()
                            {
                                found_point_2 = true;
                            }
                        }
                    }
                }
                distance_factor += 1;
            }

            let mut current_particle: (u16, u16) = closest_non_path_particle;
            while !(current_particle.0 == closest_path_particle.0
                && current_particle.1 == closest_path_particle.1)
            {
                let current_x = current_particle.0;
                let current_y = current_particle.1;
                //Checks if particle exists
                if self.grid[current_x as usize][current_y as usize]
                    != SOPSBridEnvironment::PARTICLE_LAND
                    && self.grid[current_x as usize][current_y as usize]
                        != SOPSBridEnvironment::PARTICLE_OFFLAND
                    && self.grid[current_x as usize][current_y as usize]
                        != SOPSBridEnvironment::ANCHOR
                {
                    self.grid[current_x as usize][current_y as usize] = if self.grid
                        [current_x as usize][current_y as usize]
                        == SOPSBridEnvironment::EMPTY_LAND
                    {
                        SOPSBridEnvironment::PARTICLE_LAND
                    } else {
                        SOPSBridEnvironment::PARTICLE_OFFLAND
                    };

                    self.phantom_participants.push(Particle {
                        x: current_x,
                        y: current_y,
                        onland: self.grid[current_x as usize][current_y as usize]
                            == SOPSBridEnvironment::PARTICLE_LAND,
                    });

                    self.participants.push(Particle {
                        x: current_x,
                        y: current_y,
                        onland: self.grid[current_x as usize][current_y as usize]
                            == SOPSBridEnvironment::PARTICLE_LAND,
                    });
                }

                let mut min_value: (i8, i32) = (i8::MAX, i32::MAX);
                let mut next_position: Option<(u16, u16)> = None;

                //Checks neighborhood for next particle
                for direction in SOPSBridEnvironment::directions() {
                    let checking_x: i32 = current_x as i32 + direction.0;
                    if checking_x >= 0 && checking_x < self.grid.len() as i32 {
                        let checking_y: i32 = current_y as i32 + direction.1;
                        if (checking_y >= 0 && checking_y < self.grid[0].len() as i32)
                            && !(checking_x == current_x.into()
                                && checking_y == current_y.into())
                        {
                            let opt_value =
                                opt_matrix[checking_x as usize][checking_y as usize];
                            if opt_value.1 >= 0 {
                                if (opt_value.0 < min_value.0)
                                    || (opt_value.0 == min_value.0 && opt_value.1 < min_value.1)
                                {
                                    min_value = opt_value;
                                    next_position = Some((checking_x as u16, checking_y as u16));
                                }
                            }
                        }
                    }
                }

                if let Some(next) = next_position {
                    current_particle = next;
                } else {
                    panic!("Could not find next particle in traceback.")
                }
            }

            path_particles = self.find_path_particles(anchor1_coordinates);
        }

        let decay_factor: f32 = 0.1;

        let phantom_amount: u16 = self.phantom_participants.len() as u16;
        let phantom_metric: f32 =
            std::f32::consts::E.powf(-1.0 * decay_factor as f32 * phantom_amount as f32);

        return phantom_metric.max(0.0);
    }

    /**
     * Description: Returns thee metric used to evaluate phantom particle placement.
     * Parameters:
     *  -coordinate: The coordinate being considered.
     *  -opt_matrix: The matrix being used to remember scores.
     * Return:
     *  The score to be stored within the opt matrix.
     */
    fn phantom_metric(&self, coordinate: (u16, u16), opt_matrix: &Vec<Vec<(i8, i32)>>) -> (i8, i32) {
        let mut min_value: (i8, i32) = (i8::MAX, i32::MAX);

        //Check neighborhood for pre-computed values
        for direction in SOPSBridEnvironment::directions() {
            let checking_x: i32 = coordinate.0 as i32 + direction.0;
            if checking_x >= 0 && checking_x < self.grid.len() as i32 {
                let checking_y: i32 = coordinate.1 as i32 + direction.1;
                if checking_y >= 0 && checking_y < self.grid.len() as i32 {
                    let opt_value = opt_matrix[checking_x as usize][checking_y as usize];
                    if opt_value.1 >= 0 {
                        if opt_value.0 < min_value.0 || (opt_value.0 == min_value.0 && opt_value.1 < min_value.1)
                        {
                            min_value = opt_value;
                        }
                    }
                }
            }
        }

        if self.grid[coordinate.0 as usize][coordinate.1 as usize]
            != SOPSBridEnvironment::PARTICLE_LAND
            && self.grid[coordinate.0 as usize][coordinate.1 as usize]
                != SOPSBridEnvironment::PARTICLE_OFFLAND
        {
            min_value = (min_value.0 + 1, min_value.1);
        }

        min_value = (min_value.0, min_value.1 + 1);
        return min_value;
    }

    /**
     * Description: Removes phantom particles from grid.
     */
    fn remove_phantom_participants(&mut self) {
        for i in self.phantom_participants.iter() {
            self.grid[i.x as usize][i.y as usize] =
                if self.grid[i.x as usize][i.y as usize] == SOPSBridEnvironment::PARTICLE_LAND {
                    SOPSBridEnvironment::EMPTY_LAND
                } else {
                    SOPSBridEnvironment::EMPTY_OFFLAND
                };

            if let Some(index) = self
                .participants
                .iter()
                .position(|j| j.x == i.x && j.y == i.y)
            {
                self.participants.remove(index);
            } else {
                println!("Invalid phantom at ({},{})", i.x, i.y);
                self.print_grid();
                panic!("Found particle in phantom and not participants.");
            }
        }

        self.phantom_participants = vec![];
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

        return (self.bridge_optimal_distance() as f32 / current_distance as f32).powf(3.0);
    }

    /**
     * Description:
     *  Determines fitness of current configuration based on three fitness criteria:
     *      Distance, Resources, Strength
     * Return:
     *  A f32 value between 0 and 1.
     */
    pub fn evaluate_fitness(&mut self) -> f32 {
        let strength_factor: f32 = self.strength_weight;
        let distance_factor: f32 = self.distance_weight;
        let connectivity_factor: f32 = self.connectivity_weight;
        let resource_factor: f32 = self.resource_weight;

        let total_factor: f32 =
            strength_factor + distance_factor + connectivity_factor + resource_factor;

        let mut connectivity_score: f32 = 1.0;
        let mut distance_ratio = self.calculate_distance_ratio();
        
        if distance_ratio == 0.0 {
            connectivity_score = self.add_phantom_participants();
            distance_ratio = self.calculate_distance_ratio();
        }

        let bridge_strength: f32 = self.bridge_strength();
        
        if self.phantom_participants.len() > 0 {
            self.remove_phantom_participants();
        }

        let resource_metric: f32 = self.bridge_resource();

        let return_value = f32::max(
            ((distance_factor * distance_ratio)
                + (connectivity_factor * connectivity_score)
                + (resource_factor * resource_metric)
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
     *  Runs the entirety of the simulation on the current genome.
     * Return:
     *  Returns the fitness value of the final step.
     */
    pub fn simulate(&mut self, take_snaps: bool) -> f32 {
        for step in 0..self.sim_duration {
            self.move_particles();
            if take_snaps
                && (step == (self.participants.len() as u64)
                    || step == (self.participants.len() as u64).pow(2))
                // && step % 25000 == 0
            {
                println!("Step: {}", step);
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

    fn theory_bfs(&self, elements: Vec<&(u16, u16)>)->usize {
         //Checks if l U l' forms a connected graph
         let mut closed_set: HashSet<(u16, u16)> = HashSet::new();
         let mut queue: VecDeque<(u16, u16)> = VecDeque::new();
         queue.push_back(*elements[0]);

         while queue.len() > 0 {
             if let Some(current_element) = queue.pop_front() {
                 closed_set.insert(current_element);

                 for direction in SOPSBridEnvironment::directions() {
                    let checking_x: i16 = current_element.0 as i16 + direction.0 as i16;
                     if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                        let checking_y: i16 = current_element.1 as i16 + direction.1 as i16;
                            if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)) {
                                if elements.contains(&&(checking_x as u16, checking_y as u16)) && !closed_set.contains(&(checking_x as u16, checking_y as u16)) {
                                    queue.push_back((checking_x as u16, checking_y as u16));
                                }
                            }
                     }
                 }
             } else {
                 panic!("Empty Queue!");
             }
         }

        return closed_set.len();
    }

    
    /**
     * Description:
     *  Runs the theory simulation on current grid.
     * Return:
     *  Returns the fitness value of the final step.
     */
    pub fn simulate_theory(&mut self, duration: u64, take_snaps: bool) -> f32 {
        // Bias parameters
        static LAMBDA: f64 = 4.0;
        static GAMMA: f64 = 2.0;

        let mut rng = thread_rng();
        let index_distribution = self.unfrm_par();
        let probability_distribution = Uniform::new_inclusive(0.0,1.0);
        
        for step in 0..duration {
            //Chooses particle index uniformly at random
            let participants_index = rng.sample(index_distribution);
            let offsets_values = SOPSBridEnvironment::directions();
            if let Some(current_participant) = self.participants.get(participants_index) {
                let q = rng.sample(probability_distribution);
                let mut offsets = offsets_values[rng.sample(Uniform::new(0, offsets_values.len())) as usize];
                let mut l_prime = (current_participant.x as i32 + offsets.0, current_participant.y as i32 + offsets.1);

                let mut offset_try_count = 0;
                let offset_max_count = 20;

                while  !(l_prime.0 >= 0 && (0..self.grid.len()).contains(&(l_prime.0 as usize))) 
                || !(l_prime.1 >= 0 && (0..self.grid.len()).contains(&(l_prime.1 as usize))) {
                    offsets = offsets_values[rng.sample(Uniform::new(0, offsets_values.len())) as usize];
                    l_prime = (current_participant.x as i32 + offsets.0, current_participant.y as i32 + offsets.1);
                    offset_try_count += 1;

                    if offset_try_count == offset_max_count {
                        break;
                    }
                }

                if offset_try_count == offset_max_count {
                    continue;
                }

                if self.grid[l_prime.0 as usize][l_prime.1 as usize] == SOPSBridEnvironment::EMPTY_LAND || 
                self.grid[l_prime.0 as usize][l_prime.1 as usize] == SOPSBridEnvironment::EMPTY_OFFLAND {
                    //Checks property 1
                    let mut l_neighbors: HashSet<(u16, u16)> = HashSet::new();
                    let mut l_prime_neighbors: HashSet<(u16, u16)> = HashSet::new();
                    for direction in SOPSBridEnvironment::directions() {
                        let checking_x_l: i32 = current_participant.x as i32 + direction.0 as i32;
                        let checking_x_l_prime: i32 = l_prime.0 as i32 + direction.0 as i32;
                        
                        let checking_y_l: i32 = current_participant.y as i32 + direction.1;
                        let checking_y_l_prime: i32 = l_prime.1 as i32 + direction.1;

                        //Gets l neighborhood
                        if (checking_x_l >= 0 && (0..self.grid.len()).contains(&(checking_x_l as usize))) && 
                        (checking_y_l >= 0 && (0..self.grid.len()).contains(&(checking_y_l as usize))) {
                            if self.grid[checking_x_l as usize][checking_y_l as usize] == SOPSBridEnvironment::PARTICLE_LAND ||
                            self.grid[checking_x_l as usize][checking_y_l as usize] == SOPSBridEnvironment::PARTICLE_OFFLAND ||
                            self.grid[checking_x_l as usize][checking_y_l as usize] == SOPSBridEnvironment::ANCHOR {
                                if !(checking_x_l as u16 == current_participant.x && checking_y_l as u16 == current_participant.y) {
                                    l_neighbors.insert((checking_x_l as u16, checking_y_l as u16));
                                } 
                            }
                        }
                        
                        //Checks l_prime neighborhood
                        if (checking_x_l_prime >= 0 && (0..self.grid.len()).contains(&(checking_x_l_prime as usize))) && 
                        (checking_y_l_prime >= 0 && (0..self.grid.len()).contains(&(checking_y_l_prime as usize))) {
                            if self.grid[checking_x_l_prime as usize][checking_y_l_prime as usize] == SOPSBridEnvironment::PARTICLE_LAND ||
                            self.grid[checking_x_l_prime as usize][checking_y_l_prime as usize] == SOPSBridEnvironment::PARTICLE_OFFLAND ||
                            self.grid[checking_x_l_prime as usize][checking_y_l_prime as usize] == SOPSBridEnvironment::ANCHOR {
                                if !(checking_x_l_prime as u16 == current_participant.x && checking_y_l_prime as u16 == current_participant.y) {
                                    l_prime_neighbors.insert((checking_x_l_prime as u16, checking_y_l_prime as u16));
                                }
                            }
                        }
                    }

                    let mut property1 = false;
                    let intersection = l_neighbors.intersection(&l_prime_neighbors);
                    let union = l_neighbors.union(&l_prime_neighbors);

                    if intersection.clone().count() == 1 || intersection.clone().count() == 2 {
                        let elements: Vec<&(u16, u16)> = union.clone().into_iter().collect();

                        if self.theory_bfs(elements.clone()) == elements.len() {
                            property1 = true;
                        }
                    }

                    let mut property2 = false;

                    //Checks property 2
                    if !property1 && intersection.clone().count() == 0 && l_neighbors.len() > 0 && l_prime_neighbors.len() > 0 {
                        // Checks l neighborhood completeness
                        let n_l_elements: Vec<&(u16, u16)> = (&l_neighbors).into_iter().collect();
                        let n_l_prime_elements: Vec<&(u16, u16)> = (&l_prime_neighbors).into_iter().collect();

                        property2 = self.theory_bfs(n_l_elements.clone()) == n_l_elements.len() && self.theory_bfs(n_l_prime_elements.clone()) == n_l_prime_elements.len();
                    }

                    if (property1 || property2) && l_neighbors.len() < 5 {
                        let delta_p_sigma = l_prime_neighbors.len() as f64 - l_neighbors.len() as f64;
                        
                        let mut delta_g_sigma: f64 = if !current_participant.onland {1.0} else {0.0};
                        delta_g_sigma -=  if self.grid[l_prime.0 as usize][l_prime.1 as usize] == SOPSBridEnvironment::EMPTY_OFFLAND {1.0} else {0.0};

                        //Finds the amount of disjoint graphs
                        let mut disjoint_graphs = 0;

                        let mut queue: VecDeque<(u16, u16)> = VecDeque::new();
                        let mut remaining_neighbors: HashSet<(u16, u16)> = l_neighbors.clone();

                        while remaining_neighbors.len() != 0 {
                            if queue.len() == 0 {
                                disjoint_graphs += 1;

                                let neighbors_vec: Vec<(u16, u16)> = remaining_neighbors.clone().into_iter().collect();
                                
                                queue.push_back(neighbors_vec[0]);
                                if !remaining_neighbors.remove(&(neighbors_vec[0])) {
                                    panic!("Unable to find ({},{}) in remaining neighbors.", neighbors_vec[0].0, neighbors_vec[0].1);
                                }
                            } else {
                                if let Some(current_neighbor) = queue.pop_front() {
                                    //Checks neighbors
                                    for direction in SOPSBridEnvironment::directions() {
                                        let checking_x = current_neighbor.0 as i32 + direction.0;
                                        if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                                            let checking_y = current_neighbor.1 as i32 + direction.1;
                                            if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)) {
                                                if remaining_neighbors.contains(&(checking_x as u16, checking_y as u16)) {
                                                    queue.push_back((checking_x as u16, checking_y as u16));
                                                    remaining_neighbors.remove(&(checking_x as u16, checking_y as u16));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        delta_g_sigma *= (disjoint_graphs as f64 - 1.0).max(1.0);

                        for particle in union.clone().into_iter() {
                            let mut term_value = if self.grid[particle.0 as usize][particle.1 as usize] == SOPSBridEnvironment::PARTICLE_OFFLAND { 1.0 } else { 0.0 };

                            if l_neighbors.contains(&particle) && !l_prime_neighbors.contains(&particle) {
                                let mut neighbor_count = 0;

                                for direction in SOPSBridEnvironment::directions() {
                                    let checking_x = particle.0 as i32 + direction.1;
                                    if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                                        let checking_y = particle.1 as i32 + direction.1;
                                        if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)) {
                                            if l_neighbors.contains(&(checking_x as u16, checking_y as u16)) {
                                                neighbor_count += 1;
                                            }
                                        }
                                    }
                                }
                                
                                
                                term_value *= if neighbor_count == 2 { -1.0 } else if neighbor_count == 0 { 1.0 } else { 0.0 };
                            } else if !l_neighbors.contains(&particle) && l_prime_neighbors.contains(&particle) {
                                let mut neighbor_count_with_l = 0;
                                let mut neighbor_count_with_l_prime = 0;


                                for direction in SOPSBridEnvironment::directions() {
                                    let checking_x = particle.0 as i32 + direction.0;
                                    if checking_x >= 0 && (0..self.grid.len()).contains(&(checking_x as usize)) {
                                        let checking_y = particle.1 as i32 + direction.1;
                                        if checking_y >= 0 && (0..self.grid.len()).contains(&(checking_y as usize)) {
                                            if l_neighbors.contains(&(checking_x as u16, checking_y as u16)) && !l_prime_neighbors.contains(&(checking_x as u16, checking_y as u16))  {
                                                neighbor_count_with_l += 1;
                                            } else if !l_neighbors.contains(&(checking_x as u16, checking_y as u16)) && l_prime_neighbors.contains(&(checking_x as u16, checking_y as u16))  {
                                                neighbor_count_with_l_prime += 1;
                                            }
                                        }
                                    }
                                }
                                
                                term_value *= if neighbor_count_with_l > 0 { -1.0 } else if neighbor_count_with_l_prime > 0 { 1.0 } else { 0.0 };
                            }

                            delta_g_sigma += term_value;
                        }                    


                        if q < LAMBDA.powf(delta_p_sigma) * GAMMA.powf(delta_g_sigma) {
                            //Contracts back to l
                            self.move_particle_to(participants_index, ((offsets.0 as i32), (offsets.1 as i32)));
                        }
                    }
                }
            }

            //Snapshots
            if take_snaps
                && step % 50000 == 0
            {
                println!("Step: {}", step);
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

    /**
     * Description: Measures euclidean distance between points.
     * Parameters:
     *  - point1: The first point being measured.
     *  - point2: The second point being measured.
     * Return:
     *  A f32 value representing the euclidean distance between point1 and point2.
     */
    pub fn euclidean_distance(point1: &(u16, u16), point2: &(u16, u16)) -> f32 {
        let x_diff = f32::from(point2.0) - f32::from(point1.0);
        let y_diff = f32::from(point2.1) - f32::from(point1.1);

        let distance = f32::sqrt(x_diff.powi(2) + y_diff.powi(2));
        return distance;
    }
}
