use crate::SOPSCore::aggregation_cma::SOPSEnvironmentCMA;

use super::Genome;
use rand::{distributions::Bernoulli, distributions::Uniform, rngs, Rng};
use rand_distr::{Normal, Distribution};
use rand_distr::num_traits::abs_sub;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;
use std::usize;
use std::io::Write;
use std::fs::File;
use ordered_float::OrderedFloat;
use nalgebra::*;

/*
 * Main GA class for Separation behavior (use as a model to structure and write other GA extensions for other GA's)
 * Provides basic 3 operators of the GAs and a step by step (1 step = 1 generation)
 * population generator for each step
 *  */
pub struct CmaAlgo {
    max_gen: u16,
    elitist_cnt: u16,
    population: Vec<Genome>,
    mut_rate: f64,
    granularity: u8,
    genome_cache: HashMap<[[[OrderedFloat<f64>; 4]; 3]; 4], f64>,
    perform_cross: bool,
    sizes: Vec<(u16,u16)>,
    trial_seeds: Vec<u64>,
    max_div: u32,
    random_seed: u32,
    mean: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    step_size: f64,
    p_sigma: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    covariance_matrix: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    p_c: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    weights: Vec<f64>,
    parent_number: u16,
    mu_eff: f64
}

impl CmaAlgo {

    const GENOME_LEN: u16 = 4 * 3 * 4;
    
    #[inline]
    fn rng() -> rngs::ThreadRng {
        rand::thread_rng()
    }

    #[inline]
    fn normal0_1() -> Normal<f64>{
        Normal::new(0.0, 1.0).unwrap()
    }

    #[inline]
    fn genome_init_rng(granularity: u8) -> Uniform<u8> {
        Uniform::new_inclusive(0, granularity)
    }

    #[inline]

    fn genome_prob_init_rng() -> Uniform<f64> {
        Uniform::new_inclusive(0.0, 1.0)
    }
    fn mean_init_rng(low: f32, high: f32) -> Uniform<f64> {
        Uniform::new_inclusive(low as f64, high as f64)
    }

    #[inline]
    fn unfrm_100() -> Uniform<u8> {
        Uniform::new_inclusive(1, 100)
    }

    #[inline]
    fn genome_rng(population_size: u16) -> Uniform<u16> {
        Uniform::new(0, population_size)
    }

    // fn genome_normal_rng(population_size: u16) -> Normal<f64> {
    //     Normal::new(-)
    // }

    // fn mut_val(&self) -> Normal<f64> {
    //     Normal::new(self.mut_mu, self.mut_sd).unwrap()
    // }

    #[inline]
    fn cross_pnt() -> Uniform<u16> {
        Uniform::new_inclusive(0, CmaAlgo::GENOME_LEN-1)
    }

    #[inline]
    fn mut_sign() -> Bernoulli {
        Bernoulli::new(0.3).unwrap()
    }

    /*
     * Initialize GA with given parameters and a random set of genome vectors
     *  */
    #[inline]
    pub fn init_ga(
        population_size: u16,
        max_gen: u16,
        elitist_cnt: u16,
        mut_rate: f64,
        granularity: u8,
        perform_cross: bool,
        sizes: Vec<(u16, u16)>,
        trial_seeds: Vec<u64>,
        random_seed: u32,
        search_interval: Vec<(f32, f32)>
    ) -> Self {
        let mut starting_pop: Vec<Genome> = vec![];

        for _ in 0..population_size {
            //init genome
            let mut genome: [[[f64; 4]; 3]; 4] = [[[0_f64; 4]; 3]; 4];
            for n in 0_u8..4 {
                for j in 0_u8..3 {
                    for i in 0_u8..4 {
                        genome[n as usize][j as usize][i as usize] = (CmaAlgo::rng().sample(CmaAlgo::genome_prob_init_rng()) as f64);
                    }
                }
            }
            starting_pop.push(Genome {
                string: (genome),
                fitness: (0.0),
            });
        }

        let genome_cache: HashMap<[[[OrderedFloat<f64>; 4]; 3]; 4], f64> = HashMap::new();

        // Initial CMA-ES values
        let mut mean = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.5);
        for i in 0..Self::GENOME_LEN.into() {
            mean[i] = CmaAlgo::rng().sample(CmaAlgo::mean_init_rng(search_interval[0].0, search_interval[0].1)) as f64
        }
        let mut step_size: f64 = 0.3 * (search_interval[0].1 - search_interval[0].0) as f64;
        let mut p_sigma = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0); //step size evolution path
        let mut covariance_matrix = DMatrix::from_diagonal_element(CmaAlgo::GENOME_LEN.into(), CmaAlgo::GENOME_LEN.into(), 1.0);
        let mut p_c = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0); //covariance matrix evolution path

        // Constant CMA-ES values
        let parent_number = population_size/2;

        // Sets up recombination weights
        // preliminary convex shape
        let mut preliminary_weights: Vec<f64> = vec![];
        for i in 1..(population_size + 1) {
            preliminary_weights.push(((population_size as f64 + 1.0)/2.0).ln() - (i as f64).ln());
        }

        // the variance effective selection mass for the mean
        let mu_eff = (0..parent_number).into_iter().map(|x| preliminary_weights[x as usize]).sum::<f64>().powf(2.0) / (0..parent_number).into_iter().map(|x| (preliminary_weights[x as usize]).powf(2.0)).sum::<f64>();
        
        // other values used to set up weights
        let mu_eff_neg = (0..population_size).into_iter().map(|x| preliminary_weights[x as usize]).sum::<f64>().powf(2.0) / (0..parent_number).into_iter().map(|x| (preliminary_weights[x as usize]).powf(2.0)).sum::<f64>();

        let c_one = 2.0 / ((Self::GENOME_LEN as f64 + 1.3).powf(2.0) + mu_eff); // rank-one update learning rate
        let c_mu = f64::min(1.0 - c_one, 2.0 * ((0.25 + mu_eff + 1.0/mu_eff - 2.0)/((Self::GENOME_LEN as f64 + 2.0).powf(2.0) + 2.0 * mu_eff/2.0))); // rank-mu update learning rate
        
        let alpha_mu = 1.0 + c_one / c_mu;
        let alpha_mu_eff = 1.0 + 2.0 * mu_eff_neg / (mu_eff + 2.0);
        let alpha_pos_def = (1.0 - c_one - c_mu) / (Self::GENOME_LEN as f64 * c_mu);

        // setting up recombination weights
        let mut weights: Vec<f64> = vec![];
        preliminary_weights.iter().for_each(|weight|{
            if *weight >= 0.0{
                weights.push(1.0 / preliminary_weights.iter().filter_map(|x| if *x >= 0.0 {Some(x)} else {None}).sum::<f64>() * weight);
            }
            else {
                weights.push(f64::min(alpha_mu, f64::min(alpha_mu_eff, alpha_pos_def)) / preliminary_weights.iter().filter_map(|x| if *x < 0.0 {Some(x)} else {None}).sum::<f64>().abs() * weight);
            }
        });

        CmaAlgo {
            max_gen,
            elitist_cnt,
            population: starting_pop,
            mut_rate,
            granularity,
            genome_cache,
            perform_cross,
            sizes,
            trial_seeds,
            random_seed,
            max_div: ((granularity-1) as u32)*(CmaAlgo::GENOME_LEN as u32),
            mean,
            step_size,
            p_sigma,
            covariance_matrix,
            p_c,
            weights,
            parent_number,
            mu_eff,
            
        }
    }

    // Takes a genome and returns an equivalent column vector for any matrix multiplication
    fn genome_to_column_vector(&self, genome: [[[f64; 4]; 3]; 4]) ->  Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> {
        let mut x: Vec<f64> = vec![];
        for n in 0_u8..4 {
            for j in 0_u8..3 {
                for i in 0_u8..4 {
                    x.push(genome[n as usize][j as usize][i as usize]);
                }
            }
        }
        DMatrix::from_row_iterator(CmaAlgo::GENOME_LEN.into(), 1, x.into_iter())
    }

    // Takes a column vector and returns an equivalent genome
    fn column_vector_to_genome(&self, column: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>) -> [[[f64; 4]; 3]; 4] {
        let mut genome: [[[f64; 4]; 3]; 4] = [[[0_f64; 4]; 3]; 4];
        let vector = column.column(0);
        let mut count = 0;
        for n in 0..4 {
            for i in 0..3 {
                for j in 0..4 {
                   genome[n][i][j] = vector[count];
                   count += 1;
                }
            }
        }
        genome
    }

    fn sample_new_population(&mut self) {
        let mut new_pop: Vec<Genome> = vec![];
        
        //print genomes for analysis
        let best_genome = self.population.iter().max_by(|&g1, &g2| g1.fitness.partial_cmp(&g2.fitness).unwrap()).unwrap();
        println!("Best Genome -> {best_genome:.5?}");

        // Write all the genomic data to a file
        {
            let mut buff: Vec<u8> = Vec::new();
            for genome in &self.population {
                for n in 0..4 {
                    for i in 0..3 {
                        for j in 0..4 {
                            //buff.push(genome.string[n][i][j] as u8);
                            buff.extend(genome.string[n][i][j].to_be_bytes());
                        }
                    }
                }
                buff.extend(genome.fitness.to_be_bytes());
            }

            let mut file = File::options().create(true).append(true).open(format!("./output/genomic_data_Agg_{}.log", self.random_seed)).expect("Failed to create genomic data file!");
            file.write_all(&buff).expect("Failed to append to the genomic data file!");
        }

        // Covariance decomposition
        let matrix_b = self.covariance_matrix.clone().symmetric_eigen().eigenvectors;
        let mut matrix_d = DMatrix::from_element(Self::GENOME_LEN.into(), Self::GENOME_LEN.into(), 0.0);
        for i in 0..Self::GENOME_LEN as usize{
            matrix_d[(i, i)] = self.covariance_matrix.clone().symmetric_eigen().eigenvalues[i].sqrt();                                                                                 
        }
        
        // Sampling process for each new individual in the population
        for _ in 0..self.population.len() as usize{
            
            // Equation 38 (creating z_k; the normally distributed vector)
            // Samples a new column vector from the normal distribution each sample iteration
            let mut z_k =  DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0);
            for i in 0..Self::GENOME_LEN as usize{
                z_k[(i, 0)] = CmaAlgo::rng().sample(&CmaAlgo::normal0_1());
            }

            // Equation 39 (creating y_k)
            let y_k = &matrix_b * &matrix_d * &z_k;

            // Equation 40 
            // x_k is one offspring
            let x_k = self.mean.clone() + self.step_size * &y_k;

            // clipping
            let mut x_clip = x_k;
            for i in 0..Self::GENOME_LEN as usize{
                x_clip[(i,0)] = if x_clip[(i,0)] < 0.0 
                {
                    0.0
                }  else if x_clip[(i,0)] > 1.0{
                    1.0
                } else {
                    x_clip[(i,0)]
                }
            }   

            // Changing x_k from a column vector into a genome and pushing it into new_pop vector
            new_pop.push( Genome{
                                string: self.column_vector_to_genome(x_clip),
                                fitness: 0.0,
                                } );
        }

        self.population = new_pop;

    }
    
    /*
     * Computes and returns updated mean
     * y is a vector containing column vectors, such that y = (x_i:lambda - mean) / step-size
     * */
     fn update_mean(&self, y: &Vec<Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>>) -> Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>{
        // Equation 41
        let mut y_w = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0);
        for i in 0..self.parent_number as usize {
            y_w += self.weights[i] * &y[i];
        } 

        // return updated mean values
        self.mean.clone() + 1.0 * self.step_size * y_w
    }

    /*
     * Computes and returns new step_size
     * Also updates step size evolution path
     * y is a vector containing column vectors, such that y = (x_i:lambda - mean) / step-size
     * */
     fn update_step_size(&mut self, y: &Vec<Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>>) -> f64{
        let n = Self::GENOME_LEN as f64;
        // step size constants
        let c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0);
        let d_sigma = 1.0 * 2.0 * f64::max(0.0, ((self.mu_eff - 1.0)/(n + 1.0)).sqrt() - 1.0) + c_sigma;

        // Equation 41
        let mut y_w = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0);
        for i in 0..self.parent_number as usize {
            y_w += self.weights[i] * &y[i];
        } 

        // Matrices to calcuate C^-1/2
        let matrix_b = self.covariance_matrix.clone().symmetric_eigen().eigenvectors;
        let mut matrix_d = DMatrix::from_element(Self::GENOME_LEN.into(), Self::GENOME_LEN.into(), 0.0);
        // D^-1
        for i in 0..Self::GENOME_LEN as usize{
            matrix_d[(i, i)] = 1.0 / self.covariance_matrix.clone().symmetric_eigen().eigenvalues[i].sqrt();                                                                                 
        }

        // update step size evolution path
        let evolution_path = (1.0 - c_sigma) * self.p_sigma.clone() + (c_sigma * (2.0 - c_sigma) * self.mu_eff).sqrt() * (&matrix_b * &matrix_d * &matrix_b.transpose()) * y_w;
        self.p_sigma = evolution_path.clone();        

        // compute new step size
        let step_size = self.step_size * f64::exp((c_sigma / d_sigma) * ((&evolution_path.norm() / ((n).sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n).powf(2.0))))) - 1.0));
       
        step_size
    }

    /* 
     * Performs covariance matrix adaptation and returns new covariance matrix
     * Also updates covariance matrix evolution path
     * y is a vector containing column vectors, such that y = (x_i:lambda - mean) / step-size
     * gen is current generation (used for heaviside function)
     *  */
    fn covariance_matrix_adaptation(&mut self, y: &Vec<Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>>, gen: u16) -> Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> {
        let n = Self::GENOME_LEN as f64;
        // covariance matrix adaptation constants
        let alpha_cov = 2.0;
        let c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n); // rank-one update cumulation path decay rate (not sure on value)
        let c_one = alpha_cov / ((n + 1.3).powf(2.0) + self.mu_eff); // rank-one update learning rate
        let c_mu = f64::min(1.0 - c_one, alpha_cov * ((0.25 + self.mu_eff + 1.0/self.mu_eff - 2.0)/((n + 2.0).powf(2.0) + alpha_cov * self.mu_eff/2.0))); // rank-mu update learning rate
        
        // Heaviside function
        let c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0);
        let h_sigma = if (self.p_sigma.norm() / (1.0 - (1.0 - c_sigma).powf(2.0 * (gen as f64 + 1.0))).sqrt()) < ((1.4 + 2.0/(n + 1.0)) * ((n).sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n).powf(2.0))))) {
            1.0
        } else {
            0.0
        };

        // Equation 41
        let mut y_w = DMatrix::from_element(Self::GENOME_LEN.into(), 1, 0.0);
        for i in 0..self.parent_number as usize {
            y_w += self.weights[i] * &y[i];
        } 

        // Matrices to calcuate C^-1/2
        let matrix_b = self.covariance_matrix.clone().symmetric_eigen().eigenvectors;
        let mut matrix_d = DMatrix::from_element(Self::GENOME_LEN.into(), Self::GENOME_LEN.into(), 0.0);
        // D^-1
        for i in 0..Self::GENOME_LEN as usize{
            matrix_d[(i, i)] = 1.0 / self.covariance_matrix.clone().symmetric_eigen().eigenvalues[i].sqrt();                                                                                 
        }

        // Equation 45
        // Update covariance matrix evolution path
        let evolution_path = (1.0 - c_c) * self.p_c.clone() + h_sigma * (c_c * (2.0 - c_c) * self.mu_eff).sqrt() * y_w;
        self.p_c = evolution_path.clone(); //updating covariance evolution path

        // Equation 46
        // Adjust weights for covariance matrix
        let mut covariance_weights = vec![]; 
        for i in 0..self.population.len() {
            if self.weights[i] >= 0.0 {
                covariance_weights.push(self.weights[i]);
            } else {
                covariance_weights.push(self.weights[i] * n / ((&matrix_b * &matrix_d * &matrix_b.transpose()) * &y[i]).norm().powf(2.0));
            }
        }

        // Equation 47
        // Calculate new covariance matrix
        let rank_one_update = &evolution_path * &evolution_path.transpose();

        let mut rank_mu_update = DMatrix::from_element(Self::GENOME_LEN.into(), Self::GENOME_LEN.into(), 0.0);
        for i in 0..self.population.len() {
            rank_mu_update += covariance_weights[i] * &y[i] * &y[i].transpose();
        }

        let mut new_covariance_matrix  = ((1.0 + c_one * ((1.0 - h_sigma) * c_c * (2.0 - c_c)) - c_one - c_mu * self.weights.iter().map(|x| x).sum::<f64>()) * self.covariance_matrix.clone()) + (c_one * rank_one_update)+ (c_mu * rank_mu_update) ;
        
        // clipping
        for i in 0..Self::GENOME_LEN as usize{
            for j in 0..Self::GENOME_LEN as usize{
                new_covariance_matrix[(i,j)] = if new_covariance_matrix[(i,j)] < 0.0 
                {
                    0.0
                }  else if new_covariance_matrix[(i,j)] > 1.0{
                    1.0
                } else {
                    new_covariance_matrix[(i,j)]
                }
            }
        }

        new_covariance_matrix
    }

    // mutate genome based on set mutation rate for every gene of the genome
    fn mutate_genome(&self, genome: &[[[f64; 4]; 3]; 4]) -> [[[f64; 4]; 3]; 4] {
        let mut new_genome = genome.clone();
        for n in 0..4 {
            for i in 0..3 {
                for j in 0..4 {
                    let smpl = CmaAlgo::rng().sample(&CmaAlgo::unfrm_100());
                    if smpl as f64 <= self.mut_rate * 100.0 {
                        // a random + or - mutation operation on each gene
                        let per_dir = CmaAlgo::rng().sample(&CmaAlgo::mut_sign());
                        new_genome[n][i][j] = (if per_dir {
                            genome[n][i][j] + 1.0
                        } else if genome[n][i][j] == 0.0 {
                            0.0
                        } else {
                            genome[n][i][j] - 1.0
                        })
                        .clamp(0.0, self.granularity.into());
                    }
                }
            }
        }
        new_genome
    }

    /*
     * Implements a simple single-point crossover operator with crossover point choosen at random in genome vector
     *  */
    // fn generate_offspring(&self, parent1: &[[[u8; 4]; 3]; 4], parent2: &[[[u8; 4]; 3]; 4]) -> [[[u8; 4]; 3]; 4] {
    //     let mut new_genome: [[[u8; 4]; 3]; 4] = [[[0_u8; 4]; 3]; 4];
    //     let cross_pnt = CmaAlgo::rng().sample(&CmaAlgo::cross_pnt());
    //     let mut cnt = 0;
    //     for n in 0..4 {
    //         for i in 0..3 {
    //             for j in 0..4 {
    //                 if cnt < cross_pnt {
    //                     new_genome[n][i][j] = parent1[n][i][j];
    //                 } else {
    //                     new_genome[n][i][j] = parent2[n][i][j];
    //                 }
    //                 cnt += 1; 
    //             }
    //         }
    //     }
    //     new_genome
    // }

    fn generate_offspring(&self, parent1: &[[[f64; 4]; 3]; 4], parent2: &[[[f64; 4]; 3]; 4]) -> [[[f64; 4]; 3]; 4] {
        let mut new_genome: [[[f64; 4]; 3]; 4] = [[[0_f64; 4]; 3]; 4];
        let cross_pnt_1 = CmaAlgo::rng().sample(&CmaAlgo::cross_pnt());
        let cross_pnt_2 = CmaAlgo::rng().sample(&CmaAlgo::cross_pnt());
        let lower_cross_pnt = if cross_pnt_1 <= cross_pnt_2 {cross_pnt_1} else {cross_pnt_2};
        let higher_cross_pnt = if cross_pnt_1 > cross_pnt_2 {cross_pnt_1} else {cross_pnt_2};

        let mut cnt = 0;
        for n in 0..4 {
            for i in 0..3 {
                for j in 0..4 {
                    if cnt < lower_cross_pnt {
                        new_genome[n][i][j] = parent1[n][i][j];
                    } else if cnt > lower_cross_pnt && cnt < higher_cross_pnt {
                        new_genome[n][i][j] = parent2[n][i][j];
                    } else {
                        new_genome[n][i][j] = parent1[n][i][j];
                    }
                    cnt += 1;
                }
            }
        }
        new_genome
    }

    /*
     * Performs the 3 operations (in sequence 1. selection, 2. crossover, 3. mutation) 
     * on the existing populations to generate new population
     *  */
    // fn generate_new_pop(&mut self) {
    //     let mut new_pop: Vec<Genome> = vec![];
    //     let mut selected_g: Vec<[[[u8; 4]; 3]; 4]> = vec![];
    //     let mut rank_wheel: Vec<usize> = vec![];
    //     //sort the genomes in population by fitness value
    //     self.population.sort_unstable_by(|genome_a, genome_b| {
    //         genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
    //     });

    //     //print genomes for analysis
    //     let best_genome = self.population[0];
    //     println!("Best Genome -> {best_genome:.5?}");

    //     for idx in 1..self.population.len() {
    //         println!("{y:.5?}", y = self.population[idx]);
    //     }
        
    //     //bifercate genomes
    //     for (index, genome) in self.population.iter().enumerate() {
    //         if index < self.elitist_cnt as usize {
    //             //separate out the elitist and directly pass them to next gen
    //             new_pop.push(*genome);
    //         }
    //         let genome_rank = self.population.len() - index;
    //         //create rank wheel for selection
    //         for _ in 0..genome_rank {
    //             rank_wheel.push(index);
    //         }
    //     }
    //     //perform selection and then (if perform_cross flag is set) single-point crossover
    //     let rank_wheel_rng = Uniform::new(0, rank_wheel.len());
    //     for _ in 0..(self.population.len() - self.elitist_cnt as usize) {
    //         let mut wheel_idx = CmaAlgo::rng().sample(&rank_wheel_rng);
    //         let p_genome_idx1 = rank_wheel[wheel_idx];
    //         if self.perform_cross {
    //             wheel_idx = CmaAlgo::rng().sample(&rank_wheel_rng);
    //             let p_genome_idx2 = rank_wheel[wheel_idx];
    //             selected_g.push(self.generate_offspring(
    //                 &self.population[p_genome_idx1].string,
    //                 &self.population[p_genome_idx2].string,
    //             ));
    //         } else {
    //             selected_g.push(self.population[p_genome_idx1].string); // added
    //         }
    //     }

    //     //perform mutation
    //     for idx in 0..selected_g.len() {
    //         let genome = selected_g[idx];
    //         // println!("Genome:{} mutations", idx);
    //         let mutated_g = self.mutate_genome(&genome);
    //         new_pop.push(Genome {
    //             string: mutated_g,
    //             fitness: 0.0,
    //         });
    //     }
    //     self.population = new_pop;
    // }

    /*
     * Performs the 3 operations (in sequence 1. selection, 2. crossover, 3. mutation) 
     * on the existing populations to generate new population
     *  */
     fn generate_new_pop(&mut self) {
        let mut new_pop: Vec<Genome> = vec![];
        let mut selected_g: Vec<[[[f64; 4]; 3]; 4]> = vec![];
        let mut crossed_g: Vec<[[[f64; 4]; 3]; 4]> = vec![];
        let population_size = self.population.len() as u16;
        //sort the genomes in population by fitness value
        // self.population.sort_unstable_by(|genome_a, genome_b| {
        //     genome_b.fitness.partial_cmp(&genome_a.fitness).unwrap()
        // });

        //print genomes for analysis
        let best_genome = self.population.iter().max_by(|&g1, &g2| g1.fitness.partial_cmp(&g2.fitness).unwrap()).unwrap();
        println!("Best Genome -> {best_genome:.5?}");

        // for idx in 1..self.population.len() {
        //     println!("{y:.5?}", y = self.population[idx].fitness);
        // }

        // Write all the genomic data to a file
        {
            let mut buff: Vec<u8> = Vec::new();
            for genome in &self.population {
                for n in 0..4 {
                    for i in 0..3 {
                        for j in 0..4 {
                            //buff.push(genome.string[n][i][j] as u8);
                            buff.extend(genome.string[n][i][j].to_be_bytes());
                        }
                    }
                }
                buff.extend(genome.fitness.to_be_bytes());
            }

            let mut file = File::options().create(true).append(true).open(format!("./output/genomic_data_Agg_{}.log", self.random_seed)).expect("Failed to create genomic data file!");
            file.write_all(&buff).expect("Failed to append to the genomic data file!");
        }
        
        //perform tournament selection
        for _ in 0..(population_size) {
            let genome_idx_1 = CmaAlgo::rng().sample(&CmaAlgo::genome_rng(population_size));
            let mut genome_idx_2;
            loop {
                genome_idx_2 = CmaAlgo::rng().sample(&CmaAlgo::genome_rng(population_size));
                if genome_idx_1 != genome_idx_2 {
                    break;
                }
            }
            let genome_1 = self.population[genome_idx_1 as usize];
            let genome_2 = self.population[genome_idx_2 as usize];
            if genome_1.fitness > genome_2.fitness {
                selected_g.push(genome_1.string);
            } else {
                selected_g.push(genome_2.string);
            }
        }
        
        //perform 2-point crossover
        for _ in 0..(population_size) {
            let genome_idx_1 = CmaAlgo::rng().sample(&CmaAlgo::genome_rng(population_size));
            let mut genome_idx_2;
            loop {
                genome_idx_2 = CmaAlgo::rng().sample(&CmaAlgo::genome_rng(population_size));
                if genome_idx_1 != genome_idx_2 {
                    break;
                }
            }
            let genome_1 = selected_g[genome_idx_1 as usize];
            let genome_2 = selected_g[genome_idx_2 as usize];
            crossed_g.push(self.generate_offspring(&genome_1,&genome_2));
        }

        //perform mutation
        for idx in 0..(population_size) {
            let genome = crossed_g[idx as usize];
            // println!("Genome:{} mutations", idx);
            let mutated_g = self.mutate_genome(&genome);
            new_pop.push(Genome {
                string: mutated_g,
                fitness: 0.0,
            });
        }
        self.population = new_pop;
    }

    // A single step of CMA ie. generation, where following happens in sequence
    // 1. calculate new population's fitness values
    // 2. Save each genome's fitness value based on mean fitness for 'n' eval trials
    // 3. Update mean
    // 4. Update step size
    // 5. Covariance Matrix Adaptation
    // 4. Generate new population based on these fitness values
    fn step_through(&mut self, gen: u16) -> f32 {
        let trials = self.trial_seeds.len();
        let seeds = self.trial_seeds.clone();
        let granularity = self.granularity.clone();

        // let trials_vec: Vec<((u16,u16),u64)> = self
        //     .sizes.clone()
        //     .into_iter()
        //     .zip(seeds)
        //     .flat_map(|v| std::iter::repeat(v).take(trials.into()))
        //     .collect();

        let mut trials_vec: Vec<((u16,u16),u64)> = Vec::new();

        self.sizes.iter().for_each(|size| {
            self.trial_seeds.iter().for_each(|seed| {
                trials_vec.push(((size.0,size.1),*seed));
            });
        });

        // TODO: run each genome in a separate compute node
        // TODO: use RefCell or lazy static to make the whole check and update into a single loop.
        let mut genome_fitnesses = vec![-1.0; self.population.len()];

        // check if the cache has the genome's fitness calculated
        self.population
            .iter()
            .enumerate()
            .for_each(|(idx, genome)| {
                //let genome_s = genome.string.clone();

                let mut genome_s: [[[OrderedFloat<f64>; 4]; 3]; 4] = [[[OrderedFloat(0_f64); 4]; 3]; 4];
                for n in 0..4 {
                    for i in 0..3 {
                        for j in 0..4 {
                            genome_s[n][i][j] = OrderedFloat(genome.string[n][i][j]);
                        }
                    }
                }

                match self.genome_cache.get(&genome_s) {
                    Some(fitness) => {
                        genome_fitnesses.insert(idx, *fitness);
                        return;
                    }
                    None => return,
                }
            });

        // update the genome if the value exists in the cache
        self.population
            .iter_mut()
            .enumerate()
            .for_each(|(idx, genome)| {
                if genome_fitnesses[idx] > -1.0 {
                    genome.fitness = genome_fitnesses[idx];
                }
            });

        self.population.par_iter_mut().for_each(|genome| {
            // bypass if genome has already fitness value calculated
            let genome_s = genome.string.clone();
            if gen > 0 && genome.fitness > 0.0 {
                return;
            }

            // Calculate the fitness for 'n' number of trials
            let fitness_tot: f64 = trials_vec.clone()
                .into_par_iter()
                .map(|trial| {
                    let mut genome_env = SOPSEnvironmentCMA::init_sops_env(&genome_s, trial.0.0, trial.0.1, trial.1.into(), granularity);
                    let g_fitness = genome_env.simulate(false);
                    // Add normalization of the fitness value based on optimal fitness value for a particular cohort size
                    // let max_fitness = SOPSEnvironment::aggregated_fitness(particle_cnt as u16);
                    // let g_fitness = 1; // added
                    g_fitness as f64 / (genome_env.get_max_fitness() as f64)

                    /*let g_fitness = &genome_s.iter().map(|x| {
                        x.iter().map(|y| {
                            y.iter().map(|z| {
                                z
                            }).sum::<f64>()
                        }).sum::<f64>()
                    }).sum::<f64>();
                    g_fitness / Self::GENOME_LEN as f64*/
                })
                .sum();
            
            /* Snippet to calculate Median fitness value of the 'n' trials
            // let mut sorted_fitness_eval: Vec<f64> = Vec::new();
            // fitness_trials.collect_into_vec(&mut sorted_fitness_eval);
            // sorted_fitness_eval.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // println!("Trials: {y:?}",y = sorted_fitness_eval);
            // println!("Mid: {y}",y=((trials / 2) as usize));
            // genome.fitness = sorted_fitness_eval[((trials / 2) as usize)];
            */

            let fitness_val = fitness_tot / (trials_vec.len() as f64) as f64;
            genome.fitness = fitness_val;
        });

        // populate the cache
        for idx in 0..self.population.len() {
            //let genome_s = self.population[idx].string.clone();

            let mut genome_s: [[[OrderedFloat<f64>; 4]; 3]; 4] = [[[OrderedFloat(0_f64); 4]; 3]; 4];
            for n in 0..4 {
                for i in 0..3 {
                    for j in 0..4 {
                        genome_s[n][i][j] = OrderedFloat(self.population[idx].string[n][i][j]);
                    }
                }
            }

            let genome_f = self.population[idx].fitness.clone();
            self.genome_cache.insert(genome_s, genome_f);
        }

        //avg.fitness of population
        let fit_sum = self
            .population
            .iter()
            .fold(0.0, |sum, genome| sum + genome.fitness);
        println!(
            "Avg. Fitness -> {}",
            fit_sum / (self.population.len() as f64)
        );

        // calculate population diversity
        // based on simple component wise euclidean distance squared*
        // of the genome vectors
        let mut pop_dist: Vec<f32> = vec![];
        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let genome1 = self.population[i];
                let genome2 = self.population[j];
                let mut dis_sum: u16 = 0;
                for n in 0..4 {
                    for i in 0..3 {
                        for j in 0..4 {
                            //let dis = (genome1.string[n][i][j]).abs_diff(genome2.string[n][i][j]);
                            let dis = (genome1.string[n][i][j] - genome2.string[n][i][j]).abs();
                            dis_sum += dis as u16;
                            // let genome1_prob = genome1.string[n][i][j] as f64 / (self.granularity as f64);
                            // let genome2_prob = genome2.string[n][i][j] as f64 / (self.granularity as f64);
                            // let dis = (genome1_prob - genome2_prob).abs();
                            // dis_sum += dis.powf(2.0);
                        }
                    }
                }
                // pop_dist.push(dis_sum.sqrt());
                pop_dist.push(dis_sum.into());
            }
        }
        let pop_diversity: f32 = pop_dist.iter().sum();
        let avg_pop_diversity: f32 = pop_diversity / (pop_dist.len() as f32);
        println!(
            "Population diversity -> {}",
            avg_pop_diversity / (self.max_div as f32)
        );

        // population as column vectors (x_i:lambda)
        // sort genomes by fitness value
        let mut sorted_pop = self.population.clone();
        sorted_pop.sort_by(|a, b| (&b.fitness).partial_cmp(&a.fitness).unwrap());
        // convert sorted genomes to column vectors (creating x_i:lambda)

        let mut vec_pop: Vec<Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>> = vec![]; 
        sorted_pop
            .iter()
            .for_each(|genome| {
                vec_pop.push(self.genome_to_column_vector(genome.string));
            });

        // y_i:lambda not <y>_w
        let mut y = vec![];
        for i in 0..self.population.len() {
            y.push((&vec_pop[i] - self.mean.clone())/self.step_size);
        } 
        
//      am i sorting the fitness values?
//      find the highest fitness values and then put them in array to be used by the weighted mean?


        //calculate new mean
        self.mean = self.update_mean(&y);
        //println!("Mean: {}", self.mean);

        //update step-size
        self.step_size = self.update_step_size(&y);
        //println!("Step Size: {}", self.step_size);

        //covariance matrix adaptation
        println!("eigen values: {:.5?}", self.covariance_matrix.clone().symmetric_eigen().eigenvalues);
        self.covariance_matrix = self.covariance_matrix_adaptation(&y, gen);

        // Matrices for eigendecomposition of C where C = B D^2 B^T
        //let matrix_b = self.covariance_matrix.clone().symmetric_eigen().eigenvectors;
        //let mut matrix_d = DMatrix::from_element(Self::GENOME_LEN.into(), Self::GENOME_LEN.into(), 0.0);
        //for i in 0..Self::GENOME_LEN as usize{
        //    matrix_d[(i, i)] = self.covariance_matrix.clone().symmetric_eigen().eigenvalues[i].sqrt();                                                                                 
        //}

        //generate new population
        self.sample_new_population();
        avg_pop_diversity
    }

    /*
     * The main loop of the GA which runs the full scale GA steps untill stopping criterion (ie. MAX Generations)
     * is reached
     *  */
    pub fn run_through(&mut self) {

        // Run the GA for given #. of Generations
        for gen in 0..self.max_gen {
            println!("Starting Gen:{}", gen);
            let now = Instant::now();
            self.step_through(gen);
            let elapsed = now.elapsed().as_secs();
            println!("Generation Elapsed Time: {:.2?}s", elapsed);
        }
        /*
         * Snippet to evaluate the final best genome evolved at the end of GA execution
         * TODO: Accept a parameter to run this snippet ?? Or save the best genomes to files if need be ?
        // let best_genome = self.population[0];
        // let mut best_genome_env = SOPSEnvironment::static_init(&best_genome.string);
        // best_genome_env.print_grid();
        // let g_fitness = best_genome_env.simulate();
        // best_genome_env.print_grid();
        // println!("Best genome's fitness is {}", g_fitness);
        // println!("{best_genome:?}");
         */
    }
}