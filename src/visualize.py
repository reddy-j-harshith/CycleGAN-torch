import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the data
with open('../checkpoints/training_logs.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data)

# Calculate batches from start (absolute batch number)
if 'batches_done' in df.columns:
    df['absolute_batch'] = df['batches_done']
else:
    # If no batches_done, calculate from epoch and batch
    df['absolute_batch'] = df['epoch'] * df['batch'].max() + df['batch']

# Convert ETA to hours remaining (if ETA is present)
if 'ETA' in df.columns:
    def parse_eta(eta_str):
        try:
            # Handle different ETA formats
            if 'days' in eta_str:
                parts = eta_str.split('days,')
                days = int(parts[0].strip())
                time_parts = parts[1].strip().split(':')
            else:
                days = 0
                time_parts = eta_str.split(':')
            
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2]) if len(time_parts) > 2 else 0
            
            return days * 24 + hours + minutes/60 + seconds/3600
        except:
            return np.nan
    
    df['eta_hours'] = df['ETA'].apply(parse_eta)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

# Create a figure with subplots
fig = plt.figure(figsize=(20, 16))

# 1. Loss Values Plot
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df['absolute_batch'], df['D_loss'], label='Discriminator Loss', linewidth=2)
ax1.plot(df['absolute_batch'], df['G_loss'], label='Generator Loss', linewidth=2)
ax1.set_title('Discriminator vs Generator Loss')
ax1.set_xlabel('Batches')
ax1.set_ylabel('Loss Value')
ax1.legend()
ax1.grid(True)

# 2. Component Losses Plot
ax2 = plt.subplot(2, 2, 2)
if 'GAN_loss' in df.columns:
    ax2.plot(df['absolute_batch'], df['GAN_loss'], label='GAN Loss', linewidth=2)
if 'cycle_loss' in df.columns:
    ax2.plot(df['absolute_batch'], df['cycle_loss'], label='Cycle Loss', linewidth=2)
if 'identity_loss' in df.columns:
    ax2.plot(df['absolute_batch'], df['identity_loss'], label='Identity Loss', linewidth=2)
ax2.set_title('Component Losses')
ax2.set_xlabel('Batches')
ax2.set_ylabel('Loss Value')
ax2.legend()
ax2.grid(True)

# 3. Loss Ratios Plot (useful for analyzing balance between losses)
ax3 = plt.subplot(2, 2, 3)
if all(col in df.columns for col in ['cycle_loss', 'GAN_loss']):
    ax3.plot(df['absolute_batch'], df['cycle_loss'] / df['GAN_loss'], 
             label='Cycle/GAN Ratio', linewidth=2)
if all(col in df.columns for col in ['identity_loss', 'GAN_loss']):
    ax3.plot(df['absolute_batch'], df['identity_loss'] / df['GAN_loss'], 
             label='Identity/GAN Ratio', linewidth=2)
if all(col in df.columns for col in ['D_loss', 'G_loss']):
    ax3.plot(df['absolute_batch'], df['D_loss'] / df['G_loss'], 
             label='D/G Ratio', linewidth=2)
ax3.set_title('Loss Ratios')
ax3.set_xlabel('Batches')
ax3.set_ylabel('Ratio Value')
ax3.legend()
ax3.grid(True)

# 4. ETA Plot if available
ax4 = plt.subplot(2, 2, 4)
if 'eta_hours' in df.columns:
    ax4.plot(df['absolute_batch'], df['eta_hours'], label='Estimated Time Remaining (hours)', 
             linewidth=2, color='purple')
    ax4.set_title('Training Time Remaining')
    ax4.set_xlabel('Batches')
    ax4.set_ylabel('Hours')
    ax4.grid(True)
else:
    # If no ETA, create a loss convergence plot instead
    window_size = max(1, len(df) // 20)  # 5% of data points
    df['G_loss_rolling'] = df['G_loss'].rolling(window=window_size).mean()
    df['D_loss_rolling'] = df['D_loss'].rolling(window=window_size).mean()
    
    ax4.plot(df['absolute_batch'], df['G_loss_rolling'], label='G Loss (Rolling Avg)', linewidth=2)
    ax4.plot(df['absolute_batch'], df['D_loss_rolling'], label='D Loss (Rolling Avg)', linewidth=2)
    ax4.set_title('Loss Convergence (Rolling Average)')
    ax4.set_xlabel('Batches')
    ax4.set_ylabel('Loss Value')
    ax4.legend()
    ax4.grid(True)

# Create additional advanced visualizations
fig2 = plt.figure(figsize=(20, 16))

# 5. Log-scale loss plot
ax5 = plt.subplot(2, 2, 1)
ax5.semilogy(df['absolute_batch'], df['D_loss'], label='Discriminator Loss', linewidth=2)
ax5.semilogy(df['absolute_batch'], df['G_loss'], label='Generator Loss', linewidth=2)
if 'GAN_loss' in df.columns:
    ax5.semilogy(df['absolute_batch'], df['GAN_loss'], label='GAN Loss', linewidth=2)
if 'cycle_loss' in df.columns:
    ax5.semilogy(df['absolute_batch'], df['cycle_loss'], label='Cycle Loss', linewidth=2)
if 'identity_loss' in df.columns:
    ax5.semilogy(df['absolute_batch'], df['identity_loss'], label='Identity Loss', linewidth=2)
ax5.set_title('Log-Scale Loss Values')
ax5.set_xlabel('Batches')
ax5.set_ylabel('Loss Value (log scale)')
ax5.legend()
ax5.grid(True)

# 6. Rate of change in losses
ax6 = plt.subplot(2, 2, 2)
# Calculate rate of change (derivative)
window = max(1, len(df) // 50)  # 2% of data points
df['D_loss_rate'] = df['D_loss'].diff(periods=window) / window
df['G_loss_rate'] = df['G_loss'].diff(periods=window) / window

ax6.plot(df['absolute_batch'][window:], df['D_loss_rate'][window:], 
         label='D Loss Rate of Change', linewidth=2)
ax6.plot(df['absolute_batch'][window:], df['G_loss_rate'][window:], 
         label='G Loss Rate of Change', linewidth=2)
ax6.set_title('Rate of Change in Losses')
ax6.set_xlabel('Batches')
ax6.set_ylabel('Rate of Change')
ax6.legend()
ax6.grid(True)

# 7. Epoch-wise average losses
ax7 = plt.subplot(2, 2, 3)
epoch_avg = df.groupby('epoch').agg({
    'D_loss': 'mean',
    'G_loss': 'mean'
}).reset_index()

ax7.bar(epoch_avg['epoch'] - 0.2, epoch_avg['D_loss'], width=0.4, label='Avg D Loss')
ax7.bar(epoch_avg['epoch'] + 0.2, epoch_avg['G_loss'], width=0.4, label='Avg G Loss')
ax7.set_title('Epoch-wise Average Losses')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Average Loss')
ax7.legend()
ax7.grid(True)

# 8. Loss distribution (density plot)
ax8 = plt.subplot(2, 2, 4)
sns.kdeplot(df['D_loss'], label='D Loss Distribution', ax=ax8)
# G_loss can be much larger, so consider a different scale or transformation
g_loss_scaled = df['G_loss'] / df['G_loss'].max() * df['D_loss'].max() * 2
sns.kdeplot(g_loss_scaled, label=f'G Loss (Scaled by {df["G_loss"].max() / (df["D_loss"].max() * 2):.2f})', ax=ax8)
ax8.set_title('Loss Distribution')
ax8.set_xlabel('Loss Value')
ax8.set_ylabel('Density')
ax8.legend()
ax8.grid(True)

# Add overall title and adjust layout
fig.suptitle('GAN Training Analysis - Basic Metrics', fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig2.suptitle('GAN Training Analysis - Advanced Metrics', fontsize=20)
fig2.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figures
fig.savefig('gan_training_basic_metrics.png', dpi=300, bbox_inches='tight')
fig2.savefig('gan_training_advanced_metrics.png', dpi=300, bbox_inches='tight')

# Display the plots
plt.show()

# Create an interactive dashboard with Plotly (optional but very useful)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Generator vs Discriminator Loss', 
            'Component Losses',
            'Loss Ratios', 
            'Training Convergence'
        )
    )
    
    # Add traces for each subplot
    fig3.add_trace(
        go.Scatter(x=df['absolute_batch'], y=df['D_loss'], name='D Loss'),
        row=1, col=1
    )
    fig3.add_trace(
        go.Scatter(x=df['absolute_batch'], y=df['G_loss'], name='G Loss'),
        row=1, col=1
    )
    
    if 'GAN_loss' in df.columns:
        fig3.add_trace(
            go.Scatter(x=df['absolute_batch'], y=df['GAN_loss'], name='GAN Loss'),
            row=1, col=2
        )
    if 'cycle_loss' in df.columns:
        fig3.add_trace(
            go.Scatter(x=df['absolute_batch'], y=df['cycle_loss'], name='Cycle Loss'),
            row=1, col=2
        )
    if 'identity_loss' in df.columns:
        fig3.add_trace(
            go.Scatter(x=df['absolute_batch'], y=df['identity_loss'], name='Identity Loss'),
            row=1, col=2
        )
    
    # Add loss ratios
    if all(col in df.columns for col in ['cycle_loss', 'GAN_loss']):
        fig3.add_trace(
            go.Scatter(
                x=df['absolute_batch'], 
                y=df['cycle_loss'] / df['GAN_loss'], 
                name='Cycle/GAN Ratio'
            ),
            row=2, col=1
        )
    if all(col in df.columns for col in ['D_loss', 'G_loss']):
        fig3.add_trace(
            go.Scatter(
                x=df['absolute_batch'], 
                y=df['D_loss'] / df['G_loss'], 
                name='D/G Ratio'
            ),
            row=2, col=1
        )
    
    # Add rolling averages
    window_size = max(1, len(df) // 20)
    df['G_loss_rolling'] = df['G_loss'].rolling(window=window_size).mean()
    df['D_loss_rolling'] = df['D_loss'].rolling(window=window_size).mean()
    
    fig3.add_trace(
        go.Scatter(
            x=df['absolute_batch'], 
            y=df['G_loss_rolling'], 
            name='G Loss (Rolling Avg)'
        ),
        row=2, col=2
    )
    fig3.add_trace(
        go.Scatter(
            x=df['absolute_batch'], 
            y=df['D_loss_rolling'], 
            name='D Loss (Rolling Avg)'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig3.update_layout(
        title_text="Interactive GAN Training Analysis",
        height=800,
        width=1200,
        showlegend=True,
        template="plotly_dark"
    )
    
    # Save as HTML
    fig3.write_html("gan_training_interactive.html")
    print("Interactive dashboard saved as 'gan_training_interactive.html'")
    
except ImportError:
    print("Plotly not installed. Skipping interactive visualization.")
    print("To install: pip install plotly")