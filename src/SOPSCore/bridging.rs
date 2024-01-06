use super::Particle;
use rand::SeedableRng;
use rand::{distributions::Uniform, rngs, Rng};


/*
 * Main Class for the Bridging Behaviour Expirament on SOPS grid.
 * Defines how the genome is interpreted and how much each transaction of
 * particles is derived from the genome. Also provides final SOPS grid evaluations
 * to assess the fitness score of the genome.
 */

pub struct SOPSBridEnviroment {
    grid: Vec<Vec<u8>>,
    participants: Vec<Particle>,
    anchors: Vec<Particle>,
    phenotype: [[[u8; 4]; 3]; 4],
    sim_duration: u64,
    fitness_val: f32,
    size: usize,
    max_fitness: u64,
    arena_layers: u16,
    particle_layers: u16,
    granularity: u8
}

impl SOPSBridEnviroment {
    //Defines grid space
    const EMPTY_LAND: u8 = 0;
    const PARTICLE_LAND: u8 = 1;
    const EMPTY_OFFLAND: u8 = 2;
    const PARTICLE_OFFLAND: u8 = 3;
    const ANCHOR: u8 = 4;

    //Defines neighborhood section
    const BACK: u8 = 0;
    const MID: u8 = 1;
    const FRONT: u8 = 2;

    #[inline]
    fn rng() -> rngs::ThreadRng {
        return rand::thread_rng()
    }

    #[inline]
    fn seed_rng(seed: u64) -> rngs::StdRng {
        return rand::rngs::StdRng::seed_from_u64(seed);
    }

    #[inline]
    fn move_frng() -> fastrand::Rng {
        return fastrand::Rng::new()
    }

    #[inline]
    fn grid_rng(size_s: usize, size_e: usize)->Uniform<usize> {
        return Uniform::new(size_s, size_e);
    }

    #[inline]
    fn unfrm_move() -> Uniform<u64> {
        return Uniform::<u64>::new(0, 1000);
    }

    #[inline]
    fn unfrm_dir() -> Uniform<usize> {
        return Uniform::new(0,6);
    }

    #[inline]
    fn directions() -> Vec<(i32, i32)>{
        return vec![(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1)]
    }

    fn unfrm_par(&self) -> Uniform<usize> {
        return Uniform::new(0, self.participants.len())
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
    pub fn init_sops_env(genome: &[[[u8; 4]; 3]; 4], arena_layers: u16, particle_layers: u16, seed: u64, granularity: u8) -> Self {
        let grid_size = (arena_layers * 2 + 1) as usize;
        let mut grid: Vec<Vec<u8>> = vec![vec![0; grid_size]; grid_size];
        let mut participants: Vec<Particle> = vec![];
        let mut anchors: Vec<Particle> = vec![];
        let num_particles = 6 * particle_layers * (1 + particle_layers) / 2 + 1;
        let k = 3 * particle_layers;
        let agg_edge_cnt: u64 = (k * (k + 1)).into();
        let mut grid_rng = SOPSBridEnviroment::seed_rng(seed);

        
        //Initializes grid.
        let pyramid_width_factor = 4;

        //Above pyramid.
        for i in 0..grid_size/pyramid_width_factor {
            for j in 0..grid_size {
                grid[i][j] = SOPSBridEnviroment::EMPTY_LAND;
            }
        }
        
        //Creates offland pyramid, particles, and anchors.
        for i in (grid_size/pyramid_width_factor)..grid_size {
            let pyramid_row = i - (grid_size / pyramid_width_factor);
            for j in 0..grid_size {
                let start_index: usize = ((grid_size as f32 - 1.0) / 2.0).floor() as usize - ((2 * pyramid_row) / pyramid_width_factor);
                let end_index: usize = ((grid_size as f32 - 1.0) / 2.0).ceil() as usize + ((2 * pyramid_row) / pyramid_width_factor);
                
                if j == start_index || j == end_index {
                    if i == (grid_size - 1) {
                        anchors.push(Particle {
                            x: i as u8,
                            y: j as u8,
                            color: 0
                        });

                        grid[i][j] = SOPSBridEnviroment::ANCHOR;
                    } else {
                        //Adds particpant
                        participants.push(Particle {
                            x: i as u8,
                            y: j as u8,

                            //Color = 0 when Particle is on land.
                            //Color = 1 when Particle is offland.
                            color: 0
                        });

                        grid[i][j] = SOPSBridEnviroment::PARTICLE_LAND;
                    }
                } else if j > start_index && j < end_index {
                    grid[i][j] = SOPSBridEnviroment::EMPTY_OFFLAND;
                } else {
                    grid[i][j] = SOPSBridEnviroment::EMPTY_LAND;
                }
            }
        }

        SOPSBridEnviroment {
            grid,
            participants,
            anchors,
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
            for j in 0..self.grid[i].len() {
                print!(" {} ", self.grid[i][j])
            }
            println!("")
        }
    }

    /**
     * Description:
     *  TODO
     * Parameters:
     *  TODO
     * Return:
     *  TODO
     */
    fn particle_move_possible(&self, particle_idx: usize, direction: (i32, i32)) -> bool {
        todo!();
    }

    /**
     * Description:
     *  TODO
     * Parameters:
     *  TODO
     * Return:
     *  TODO
     */
    fn move_particle_to(&mut self, particle_idx: usize, direction: (i32, i32)) -> bool {
        todo!();
    }

    /**
     * Description:
     *  TODO
     * Parameters:
     *  TODO
     * Return:
     *  TODO
     */
    fn move_particle(&mut self, cnt: usize) {
        todo!();
    }

    /**
     * Description: 
     *  Finds the unit vector from the current_point to anchor. Uses round to determine which direction on the grid is closest.
     * Return:
     *  A (i8, i8) tuple representing the unit vector (rounded) in the direction of the anchor point.
     */
    fn direction_to(&self, current_point: &(u8, u8), anchor: &(u8, u8))->(i8, i8) {
        let vector: (i8, i8) = (anchor.0 as i8 - current_point.0 as i8, anchor.1 as i8 - current_point.1 as i8);
        let vector_magnitude: f32 = (((vector.0).pow(2) + (vector.1).pow(2)) as f32).sqrt();

        return ((vector.0 as f32 / vector_magnitude).round() as i8, (vector.1 as f32 / vector_magnitude).round() as i8);
    }

    /**
     * Description: 
     *  Rotates unit_vector according to rotation_matrix.
     * Return:
     *  A (i8, i8) tuple representing the rotated unit vector (ceiled).
     */
    fn rotate_unit_vector(&self, rotation_matrix: &Vec<Vec<f32>>, unit_vector: (i8, i8))-> (i8, i8) {
        assert!(rotation_matrix[0].len() == 2, "Cannot multiply matrices of these sizes together");

        let unit_vector_as_vec = vec![unit_vector.0, unit_vector.1];

        let mut rotated_unit_vector: Vec<f32> = vec![0.0; unit_vector_as_vec.len()];

        for i in 0..rotation_matrix.len() {
            for j in 0..rotation_matrix[i].len() {
                rotated_unit_vector[i] += rotation_matrix[i][j] * unit_vector_as_vec[j] as f32;
            }
        }
        
        return (rotated_unit_vector[0].round() as i8, rotated_unit_vector[1].round() as i8);
    }

    /**
     * Description: 
     *  Measures the distance between the two anchor points within the grid.
     * Return:
     *  A u32 value representing the amount of particles between the anchors. Returns 0 if the anchors are disconnected.
     */
    fn bridge_distance(&self) -> u32 {
        assert!(self.anchors.len() == 2, "Distance function is made for 2 anchors!");

        let anchor2: (u8, u8) = (self.anchors[1].x, self.anchors[1].y);

        let mut current_particle: (u8, u8) = (self.anchors[0].x, self.anchors[0].y);
        let mut bridge_length: u32 = 0;


        let root_two_over_two = 0.7071067;
        let rotation_matrices: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![1.0,0.0], vec![0.0,1.0]], //0
            vec![vec![root_two_over_two, -root_two_over_two], vec![root_two_over_two, root_two_over_two]], //pi/4
            vec![vec![root_two_over_two, root_two_over_two], vec![-root_two_over_two, root_two_over_two]], //7pi/4
            vec![vec![0.0, -1.0], vec![1.0, 0.0]], //pi/2
            vec![vec![0.0, 1.0], vec![-1.0, 0.0]], //3pi/2
            vec![vec![-root_two_over_two, -root_two_over_two], vec![root_two_over_two, -root_two_over_two]], //3pi/4
            vec![vec![-root_two_over_two, root_two_over_two], vec![-root_two_over_two, -root_two_over_two]], //5pi/4
            vec![vec![-1.0, 0.0], vec![0.0, -1.0]], //pi
        ];

        while current_particle != anchor2 {
            //Calculates direction from current_particle to anchor2
            //Favors anchor2 direction when selecting next point
            let unit_vector: (i8, i8) = self.direction_to(&current_particle, &anchor2);
            let old_particle: (u8, u8) = current_particle;

            bridge_length += 1;

            for i in 0..rotation_matrices.len() {
                let rotated_unit_vector = self.rotate_unit_vector(&rotation_matrices[i], unit_vector);

                let check_x:i8 = current_particle.0 as i8 + rotated_unit_vector.0;
                let check_y:i8 = current_particle.1 as i8 + rotated_unit_vector.1;                

                if check_x >= 0 && check_y >= 0 {
                    if self.grid[check_x as usize][check_y as usize] == SOPSBridEnviroment::PARTICLE_LAND || self.grid[check_x as usize][check_y as usize] == SOPSBridEnviroment::PARTICLE_OFFLAND {
                        current_particle = (check_x as u8, check_y as u8);
                        break;
                    }
                } else {
                    println!("Out of bounds at ({},{}) for ({},{}).", check_x, check_y, current_particle.0, current_particle.1);
                }
            }

            if current_particle == old_particle {
                //Returns 0 if disconnected or an island (local maxima).
                return 0;
            }
        }

        return bridge_length;
    }

    /**
     * Description:
     *  Caluclates optimal distance from anchor1 to anchor2.
     * Return:
     *  A u32 value representing the amount of particles between.
     */
    fn bridge_optimal_distance(&self) -> u32 {
        assert!(self.anchors.len() == 2, "Optimal distance function is made for 2 anchors!");

        let anchor2_coordinates: (u8, u8) = (self.anchors[1].x, self.anchors[1].y);

        let mut current_coordinates: (u8, u8) = (self.anchors[0].x, self.anchors[0].y);
        let mut bridge_length: u32 = 0;

        while current_coordinates != anchor2_coordinates {
            bridge_length += 1;

            //Determines direction of anchor2_coordinate
            let unit_vector: (i8, i8) = self.direction_to(&current_coordinates, &anchor2_coordinates);
            let check_x:i8 = current_coordinates.0 as i8 + unit_vector.0;
            let check_y:i8 = current_coordinates.1 as i8 + unit_vector.1;

            if check_x >= 0 && check_y >= 0 {
                current_coordinates = (check_x as u8, check_y as u8);
            } else {
                println!("Out of bounds at ({},{}) for ({},{}).", check_x, check_y, current_coordinates.0, current_coordinates.1);
            }
        }

        return bridge_length;
    }

    /**
     * Description: 
     *  TODO
     * Return:
     *  TODO
     */
    fn bridge_resource(&self) -> f32 {
        todo!();
    }

    /**
     * Description: 
     *  TODO
     * Return:
     *  TODO
     */
    fn bridge_strength(&self) -> f32 {
        todo!();
    }

    /**
     * Description:
     *  Determines fitness of current configuration based on three fitness criteria: 
     *      Distance, Resources, Strength
     * Return:
     *  A f32 value between 0 and 1.
     */
    pub fn evaluate_fitness(&self) -> f32 {
        //Calculates distance
        let distance_ratio: f32 = self.bridge_distance() as f32/self.bridge_optimal_distance() as f32;
        if distance_ratio == 0.0 {
            return 0.0;
        }

        let bridge_resource: f32 = self.bridge_resource();
        let bridge_strength: f32 = self.bridge_strength();

        return (distance_ratio + bridge_resource + bridge_strength) / 3.0;
    }

    /**
     * Description:
     *  TODO
     * Parameters:
     *  TODO
     * Return:
     *  TODO
     */
    pub fn simulate(&mut self, take_snaps: bool) -> u32 {
        todo!();
    }

    /*
    * Description:
    *   TODO
    * Return:
    *   TODO
    */
    pub fn get_max_fitness(&self) -> u64 {
        todo!();
    }

    /*
    * Description:
    *   TODO
    * Return:
    *   TODO
    */
    pub fn get_participant_cnt(&self) -> usize {
        todo!();
    }
}