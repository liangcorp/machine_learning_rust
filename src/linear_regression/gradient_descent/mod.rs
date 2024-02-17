//! Gradient descent with multiple features
//!
//! This crate is a collection of functions to perform
//! calculation of gradient descent

use std::io;
use std::io::{Error, ErrorKind};

/// # Gradient descent
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
///
///
pub fn get_thetas(
    x_mtrx: &[Vec<f64>],
    y_vec: &[f64],
    alpha: f32,
    theta: &mut [f64],
    iterations: u32,
) -> Result<Box<Vec<f64>>, io::Error> {
    let num_train = y_vec.len(); // no of training sets
    let num_feat = theta.len();

    let mut sum: f64;
    let mut h_x = vec![0.0; num_train];

    if x_mtrx.len() != num_train {
        return Err(Error::new(ErrorKind::Other, "Mis-matching training sets"));
    }

    // Convert Vec<Vec<f64>> to &[&[f64]]
    // to speeds up the execution by a little
    let mut x_vec_slice: Vec<&[f64]> = Vec::with_capacity(num_train);

    for x_row in x_mtrx.iter().take(num_train) {
        x_vec_slice.push(&x_row[..]);
    }

    let x_slice = &x_vec_slice[..];

    for _ in 0..iterations {
        // Shadow h_x from vec to vec slice
        // speeds up the execution a bit
        let h_x = &mut h_x[..];
        for (i, x_row) in x_slice[..].iter().enumerate().take(num_train) {
            sum = 0.0;
            for j in 0..num_feat {
                sum += theta[j] * x_row[j];
            }

            h_x[i] = sum;
        }

        for (j, t) in theta.iter_mut().enumerate().take(num_feat) {
            sum = 0.0;

            for i in 0..num_train {
                sum += (h_x[i] - y_vec[i]) * x_slice[i][j];
            }

            *t -= alpha * sum / num_train as f64;
        }
    }

    Ok(Box::new(theta.to_vec()))
}
