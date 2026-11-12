import torch
import gc
import logging
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()

def get_vram_usage() -> tuple:
    """
    Returns a tuple of (allocated_mb, reserved_mb, total_gb) of GPU memory.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 2) # MB
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)   # MB
    
    # Get total device memory
    device_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gb = device_properties.total_memory / (1024 ** 3) # GB
    
    return allocated, reserved, total_gb

def print_vram_report(stage: str = "Status"):
    """
    Prints a rich VRAM usage report.
    """
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Skipping VRAM report.[/yellow]")
        return
        
    allocated, reserved, total = get_vram_usage()
    allocated_gb = allocated / 1024
    reserved_gb = reserved / 1024
    
    content = (
        f"[bold]Stage:[/bold] {stage}\n"
        f"[bold]Allocated VRAM:[/bold] {allocated_gb:.2f} GB ({allocated:.1f} MB)\n"
        f"[bold]Reserved VRAM:[/bold] {reserved_gb:.2f} GB ({reserved:.1f} MB)\n"
        f"[bold]Total GPU VRAM:[/bold] {total:.2f} GB"
    )
    
    # Use green, yellow, or red based on usage percentage
    pct = allocated_gb / total if total > 0 else 0
    if pct > 0.90:
        color = "red"
    elif pct > 0.75:
        color = "yellow"
    else:
        color = "green"
        
    console.print(Panel(content, title="[bold]GPU VRAM Report[/bold]", border_style=color, expand=False))

def clear_gpu_memory():
    """
    Forces garbage collection and clears PyTorch's internal VRAM cache.
    """
    logger.info("Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.info("GPU memory cleared.")

class VRAMTracker:
    """
    Context manager to track VRAM usage before and after block execution.
    """
    def __init__(self, block_name: str):
        self.block_name = block_name
        
    def __enter__(self):
        clear_gpu_memory()
        logger.info(f"--- Entering block: {self.block_name} ---")
        print_vram_report(f"Before {self.block_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"--- Exiting block: {self.block_name} ---")
        print_vram_report(f"After {self.block_name}")
        clear_gpu_memory()
