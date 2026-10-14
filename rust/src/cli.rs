//! Command-line options shared by the binaries: `--key value` pairs.
//!
//! Sizes: `--L 6` runs one size; `--lmin 4 --lmax 8` runs a range (not both forms at once).
//! Unknown keys are an error, so a mistyped option is not silently ignored.

use std::collections::HashMap;
use std::ops::RangeInclusive;

pub struct Options {
    values: HashMap<String, String>,
}

impl Options {
    /// Parses the command line; `known` lists the accepted keys (without `--`).
    pub fn parse(known: &[&str]) -> Self {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let values: HashMap<String, String> = argv
            .chunks(2)
            .map(|pair| match pair {
                [key, value] if key.starts_with("--") => (key[2..].to_string(), value.clone()),
                _ => panic!("options are given as --key value, got {pair:?}"),
            })
            .collect();
        for key in values.keys() {
            assert!(known.contains(&key.as_str()), "unknown option --{key}; accepted: --{}", known.join(", --"));
        }
        Options { values }
    }

    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn f64(&self, key: &str, default: f64) -> f64 {
        self.values
            .get(key)
            .map_or(default, |s| s.parse().unwrap_or_else(|_| panic!("cannot parse --{key} {s}")))
    }

    pub fn usize(&self, key: &str, default: usize) -> usize {
        self.values
            .get(key)
            .map_or(default, |s| s.parse().unwrap_or_else(|_| panic!("--{key} needs a non-negative integer, got {s}")))
    }

    /// Comma-separated numbers, e.g. `--gammas 0.01,0.02`.
    pub fn list(&self, key: &str) -> Option<Vec<f64>> {
        self.values.get(key).map(|s| {
            s.split(',')
                .map(|x| x.trim().parse().unwrap_or_else(|_| panic!("--{key}: comma-separated numbers, got {s}")))
                .collect()
        })
    }

    /// System sizes: `--L` alone, or `--lmin`/`--lmax` with the given defaults.
    pub fn sizes(&self, lmin: usize, lmax: usize) -> RangeInclusive<usize> {
        if self.has("L") {
            assert!(!self.has("lmin") && !self.has("lmax"), "give either --L or --lmin/--lmax");
            let l = self.usize("L", 0);
            l..=l
        } else {
            self.usize("lmin", lmin)..=self.usize("lmax", lmax)
        }
    }
}
