//! ZeroLang CLI - The first language for machines, by machines

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use capnp::message::ReaderOptions;
use capnp::serialize;
use clap::{Parser, Subcommand, ValueEnum};

use zerolang::zero_capnp::{graph, node, proof};
use zerolang::{stdlib, verify_graph, RuntimeGraph, VerifyOptions, VM};

/// ZeroLang CLI - The first language for machines, by machines
#[derive(Parser)]
#[command(name = "zero")]
#[command(about = "Agent-to-Agent programming language toolchain", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a Zero graph file (.0)
    Generate {
        /// Output file path
        output: PathBuf,
        /// Type of graph to generate
        #[arg(short, long, default_value = "hello")]
        graph_type: GraphType,
    },
    /// Execute a Zero graph file using the 0-VM
    Execute {
        /// Input file path
        input: PathBuf,
        /// Skip verification (unsafe mode)
        #[arg(long)]
        r#unsafe: bool,
    },
    /// Inspect a Zero graph file (human-readable debug view)
    Inspect {
        /// Input file path
        input: PathBuf,
    },
    /// Verify a Zero graph file (check hashes, proofs, structure)
    Verify {
        /// Input file path
        input: PathBuf,
        /// Skip hash verification
        #[arg(long)]
        skip_hash: bool,
        /// Skip halting proof requirement
        #[arg(long)]
        skip_halting: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum GraphType {
    /// The Genesis Block - Hello World embedding
    Hello,
    /// Simple math: 1.0 + 2.0 = 3.0
    Math,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Generate { output, graph_type } => {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();

            let message = match graph_type {
                GraphType::Hello => stdlib::generate_hello_world(timestamp)?,
                GraphType::Math => stdlib::generate_simple_math(timestamp)?,
            };

            let file = File::create(output)?;
            let mut writer = BufWriter::new(file);
            serialize::write_message(&mut writer, &message)?;

            println!("Generated: {}", output.display());
            match graph_type {
                GraphType::Hello => {
                    println!("  Type: Genesis Block (Hello World)");
                    println!("  Nodes: 1 (Constant)");
                    println!("  Output: Tensor<768> with confidence 1.0");
                }
                GraphType::Math => {
                    println!("  Type: Simple Math (1.0 + 2.0)");
                    println!("  Nodes: 3 (2 Constants + 1 Add Operation)");
                    println!("  Expected Output: 3.0");
                }
            }
        }
        Commands::Execute { input, r#unsafe } => execute_graph(input, *r#unsafe)?,
        Commands::Inspect { input } => inspect_graph(input)?,
        Commands::Verify {
            input,
            skip_hash,
            skip_halting,
        } => verify_graph_cmd(input, *skip_hash, *skip_halting)?,
    }

    Ok(())
}

// ============================================================================
// EXECUTOR
// ============================================================================

/// Execute a Zero graph file using the 0-VM
fn execute_graph(input: &PathBuf, unsafe_mode: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading: {}", input.display());

    // Load the graph
    let graph = RuntimeGraph::load_from_file(input)?;

    println!("  Protocol Version: {}", graph.version);
    println!("  Nodes: {}", graph.node_count());

    // Verify before execution (unless unsafe mode)
    if !unsafe_mode {
        let options = VerifyOptions {
            verify_hashes: true,
            require_halting_proof: false, // Relaxed for now, will be strict in Phase B
            verify_shape_proofs: false,   // Not implemented yet
        };
        match verify_graph(&graph, &options) {
            Ok(result) => {
                println!(
                    "  Verification: PASSED ({} nodes, {} hash checks)",
                    result.nodes_verified, result.hash_checks_passed
                );
            }
            Err(e) => {
                println!("  Verification: FAILED - {}", e);
                println!("\n  Use --unsafe to skip verification (not recommended)");
                return Err(Box::new(e));
            }
        }
    } else {
        println!("  Verification: SKIPPED (unsafe mode)");
    }

    // Create VM (use fuel from halting proof if present)
    let mut vm = VM::from_graph(&graph);
    let outputs = vm.execute(&graph)?;

    println!("\n  Execution Statistics:");
    println!("    Operations: {}", vm.ops_executed());
    println!("    Remaining Fuel: {}", vm.remaining_fuel());

    println!("\n  Outputs ({}):", outputs.len());
    for (i, tensor) in outputs.iter().enumerate() {
        println!("    [{}] Shape: {:?}", i, tensor.shape);
        println!("        Confidence: {:.2}%", tensor.confidence * 100.0);

        if tensor.is_scalar() {
            println!("        Value: {}", tensor.as_scalar());
        } else if tensor.numel() <= 10 {
            println!("        Data: {:?}", tensor.data);
        } else {
            // For large tensors, show preview
            let preview: Vec<f32> = tensor.data.iter().take(5).copied().collect();
            println!(
                "        Data: {:?}... ({} elements)",
                preview,
                tensor.numel()
            );

            // Try to decode "HELLO" if it looks like an embedding
            if tensor.shape == vec![768] && tensor.data.len() >= 5 {
                let decoded: String = (0..5)
                    .map(|i| (tensor.data[i] * 255.0) as u8 as char)
                    .collect();
                println!("        Decoded: \"{}\"", decoded);
            }
        }
    }

    println!("\nExecution complete.");
    Ok(())
}

// ============================================================================
// INSPECTOR
// ============================================================================

/// Inspect a Zero graph file (detailed debug view)
fn inspect_graph(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    let reader = BufReader::new(file);
    let message_reader = serialize::read_message(reader, ReaderOptions::new())?;
    let graph_reader = message_reader.get_root::<graph::Reader>()?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ZERO GRAPH INSPECTOR                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("File: {}", input.display());
    println!("Protocol Version: {}", graph_reader.get_version());
    println!();

    // Metadata
    let metadata = graph_reader.get_metadata();
    println!("┌─ METADATA ─────────────────────────────────────────────────────");
    if let Ok(created_by) = metadata.get_created_by() {
        println!("│ Created By: {}", String::from_utf8_lossy(created_by));
    }
    println!("│ Created At: {}", metadata.get_created_at());
    if let Ok(desc) = metadata.get_description() {
        println!("│ Description: {}", String::from_utf8_lossy(desc));
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();

    // Nodes
    let nodes = graph_reader.get_nodes()?;
    println!(
        "┌─ NODES ({}) ──────────────────────────────────────────────────",
        nodes.len()
    );
    for (i, node) in nodes.iter().enumerate() {
        let node_id = node.get_id()?;
        let hash = hex::encode(node_id.get_hash()?);

        print!("│ [{}] ", i);
        print!("Hash: {}... ", &hash[..16]);

        match node.which()? {
            node::Constant(tensor_reader) => {
                let tensor = tensor_reader?;
                let shape: Vec<u32> = tensor.get_shape()?.iter().collect();
                let data = tensor.get_data()?;
                if shape == vec![1] && data.len() == 1 {
                    println!(
                        "CONSTANT Scalar({}) conf={:.2}",
                        data.get(0),
                        tensor.get_confidence()
                    );
                } else {
                    println!(
                        "CONSTANT Tensor<{:?}> conf={:.2}",
                        shape,
                        tensor.get_confidence()
                    );
                }
            }
            node::Operation(op) => {
                println!(
                    "OPERATION {:?} inputs={}",
                    op.get_op()?,
                    op.get_inputs()?.len()
                );
            }
            node::External(ext) => {
                println!("EXTERNAL uri={:?}", ext.get_uri()?);
            }
            node::Branch(br) => {
                println!("BRANCH threshold={:.2}", br.get_threshold());
            }
        }
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();

    // Proofs
    let proofs = graph_reader.get_proofs()?;
    println!(
        "┌─ PROOFS ({}) ──────────────────────────────────────────────────",
        proofs.len()
    );
    for (i, proof) in proofs.iter().enumerate() {
        print!("│ [{}] ", i);
        match proof.which()? {
            proof::Halting(h) => {
                println!(
                    "HALTING max_steps={} fuel={}",
                    h.get_max_steps(),
                    h.get_fuel_budget()
                );
            }
            proof::ShapeValid(_) => println!("SHAPE_VALID"),
            proof::Signature(_) => println!("SIGNATURE"),
            proof::None(()) => println!("NONE (unsafe)"),
        }
    }
    println!("└────────────────────────────────────────────────────────────────");
    println!();

    // Entry point
    let entry = graph_reader.get_entry_point()?;
    println!("Entry Point: {}...", &hex::encode(entry.get_hash()?)[..16]);

    // Outputs
    let outputs = graph_reader.get_outputs()?;
    println!("Outputs: {}", outputs.len());

    println!();
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

// ============================================================================
// VERIFIER
// ============================================================================

/// Verify a Zero graph file
fn verify_graph_cmd(
    input: &PathBuf,
    skip_hash: bool,
    skip_halting: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ZERO GRAPH VERIFIER                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("File: {}", input.display());

    // Load the graph
    let graph = RuntimeGraph::load_from_file(input)?;

    println!("Protocol Version: {}", graph.version);
    println!("Total Nodes: {}", graph.node_count());
    println!();

    // Set up verification options
    let options = VerifyOptions {
        verify_hashes: !skip_hash,
        require_halting_proof: !skip_halting,
        verify_shape_proofs: false, // Not implemented yet
    };

    println!("┌─ VERIFICATION OPTIONS ─────────────────────────────────────────");
    println!(
        "│ Hash verification: {}",
        if options.verify_hashes { "ON" } else { "OFF" }
    );
    println!(
        "│ Halting proof required: {}",
        if options.require_halting_proof {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "│ Shape proof verification: {}",
        if options.verify_shape_proofs {
            "ON"
        } else {
            "OFF"
        }
    );
    println!("└─────────────────────────────────────────────────────────────────");
    println!();

    // Run verification
    match verify_graph(&graph, &options) {
        Ok(result) => {
            println!("┌─ VERIFICATION RESULT ──────────────────────────────────────────");
            println!("│ Status: ✓ PASSED");
            println!("│ Nodes verified: {}", result.nodes_verified);
            println!("│ Hash checks: {}", result.hash_checks_passed);
            if let Some(halting) = result.halting_proof {
                println!(
                    "│ Halting proof: max_steps={}, fuel={}",
                    halting.max_steps, halting.fuel_budget
                );
            }
            println!("└─────────────────────────────────────────────────────────────────");
            println!();
            println!("Graph is valid and safe to execute.");
        }
        Err(e) => {
            println!("┌─ VERIFICATION RESULT ──────────────────────────────────────────");
            println!("│ Status: ✗ FAILED");
            println!("│ Error: {}", e);
            println!("└─────────────────────────────────────────────────────────────────");
            println!();
            println!("Graph is NOT safe to execute.");
            return Err(Box::new(e));
        }
    }

    Ok(())
}
