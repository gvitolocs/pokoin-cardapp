use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn main() {
    let revision = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CARD_IMAGES_REVISION={revision}");
    // Worktrees have their own HEAD, but branch refs live in the common git directory.
    for name in [Some("HEAD".into()), git(&["symbolic-ref", "-q", "HEAD"])]
        .into_iter()
        .flatten()
    {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
