//! # Read data from file and store the value into vectors
//!
use std::fs::File;
use std::io::{self, BufRead, Error, ErrorKind};
use std::path::Path;

type DoubleVecF64 = Vec<Vec<f32>>;

pub fn get_data(path: &Path) -> Result<(DoubleVecF64, Vec<f32>, Vec<f32>), io::Error> {
    let lines = match File::open(path) {
        Ok(file) => io::BufReader::new(file).lines(),
        Err(ref error) if error.kind() == ErrorKind::NotFound => {
            return Err(Error::new(ErrorKind::NotFound, "File not found"));
        }
        Err(error) if error.kind() == ErrorKind::PermissionDenied => {
            return Err(Error::new(ErrorKind::PermissionDenied, "Permission denied"));
        }
        Err(_) => {
            return Err(Error::new(ErrorKind::Other, "Can not open file."));
        }
    };

    let mut y: Vec<f32> = vec![];
    let mut v: Vec<String> = vec![];

    // Read the file line by line
    // split each line by the last ',' into two vectors of v and y
    for line in lines {
        if let Some(data_tuple) = line.unwrap().rsplit_once(',') {
            v.push(data_tuple.0.to_string());
            y.push(data_tuple.1.parse::<f32>().expect("Failed"));
        }
    }

    let mut tmp: Vec<Vec<&str>> = vec![];

    for i in v.iter() {
        tmp.push(i.split(',').collect::<Vec<&str>>());
    }

    let mut x: Vec<Vec<f32>> = vec![];
    let mut flatten_x: Vec<f32> = vec![];

    for i in tmp.iter() {
        let mut tmp_f32: Vec<f32> = vec![1.0];

        for j in i.iter().map(|e| e.to_string().parse::<f32>()) {
            tmp_f32.push(j.unwrap());
        }
        x.push(tmp_f32.to_vec());
        flatten_x = [&flatten_x[..], &tmp_f32[..]].concat();
    }

    Ok((x, flatten_x, y))
}
