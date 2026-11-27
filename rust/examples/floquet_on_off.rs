//! Pulsed depolarization on the L=12 scar. Pass the window length in Floquet cycles
//! (one third of the continuous-on first-crossing cycle). Add "off-first" to start off
//! and alternate every window until crossing. CSV rows label the preceding cycle.
use std::io::{self, BufWriter, Write};

use open_exact_scars::floquet::{per_site, product_state, Floquet};
use open_exact_scars::model::Params;
use open_exact_scars::noise::Channel;
use open_exact_scars::trajectory::Rng;

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let window: usize = args
        .next()
        .expect("window in cycles")
        .parse()
        .expect("integer window");
    assert!(window > 0);
    let off_first = match args.next().as_deref() {
        None => false,
        Some("off-first") => true,
        _ => panic!("expected off-first or no protocol argument"),
    };
    let l = 12;
    let dt = 0.1;
    let gamma = 0.025;
    let params = Params::random_c(l, &mut Rng::new(1000 + l as u64));
    let on = Floquet::new(&params, &Channel::Xyz, gamma, dt);
    let off = Floquet::new(&params, &Channel::Xyz, 0.0, dt);
    let mut rho = product_state(&vec![[-1.0, 0.0, 0.0]; l]);
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "cycle,t,gamma_previous,mx,my,m_perp")?;
    writeln!(out, "0,0,,-1,0,1")?;
    for n in 1.. {
        let phase = (n - 1) / window;
        let off_window = if off_first {
            phase % 2 == 0
        } else {
            phase == 1 || phase == 3
        };
        let (circuit, rate) = if off_window {
            (&off, 0.0)
        } else {
            (&on, gamma)
        };
        circuit.step(&mut rho);
        let [mx, my, _] = per_site(&rho, l).map(|values| values.iter().sum::<f64>() / l as f64);
        let amplitude = mx.hypot(my);
        writeln!(
            out,
            "{n},{:.6},{rate},{mx:.13e},{my:.13e},{amplitude:.13e}",
            n as f64 * dt
        )?;
        out.flush()?;
        if amplitude <= 1.0 / std::f64::consts::E {
            break;
        }
    }
    Ok(())
}
