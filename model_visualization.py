import graphviz


def create_model_visualization():
    # Model hyperparameters
    DEPTHS = [2, 2, 2]
    CHANNELS = [64, 128, 256]
    ATTN_DEPTHS = [False, True, True]
    
    # Create visualization
    dot = graphviz.Digraph(comment='UNet Architecture')
    dot.attr(rankdir='TB')
    
    # Input nodes
    dot.node('input', 'Input\n(B, 64, H, W)')
    dot.node('cond', 'Condition\n(B, 16)')
    
    # Encoder path
    prev_node = 'input'
    block_configs = zip(DEPTHS, CHANNELS, ATTN_DEPTHS)
    for i, (depth, channels, has_attn) in enumerate(block_configs):
        with dot.subgraph(name=f'cluster_down_{i}') as s:
            s.attr(label=f'Down Block {i}')
            
            # ResBlocks
            for j in range(depth):
                block_name = f'down_{i}_res_{j}'
                label = f'ResBlock\n({channels} ch)'
                
                if has_attn and j == depth-1:
                    label += '\n+ Attention'
                s.node(block_name, label)
                
                # Connect blocks
                if j == 0:
                    dot.edge(prev_node, block_name)
                else:
                    prev_block = f'down_{i}_res_{j-1}'
                    s.edge(prev_block, block_name)
            
            # Downsample
            if i < len(DEPTHS)-1:
                down_name = f'down_{i}_sample'
                s.node(down_name, 'Downsample\n÷2')
                s.edge(f'down_{i}_res_{depth-1}', down_name)
                prev_node = down_name
            else:
                prev_node = f'down_{i}_res_{depth-1}'
    
    # Middle block
    with dot.subgraph(name='cluster_middle') as s:
        s.attr(label='Middle Block')
        s.node('mid_1', f'ResBlock\n({CHANNELS[-1]} ch)\n+ Attention')
        s.node('mid_2', f'ResBlock\n({CHANNELS[-1]} ch)\n+ Attention')
        dot.edge(prev_node, 'mid_1')
        s.edge('mid_1', 'mid_2')
        prev_node = 'mid_2'
    
    # Decoder path
    decoder_configs = zip(
        reversed(DEPTHS),
        reversed(CHANNELS),
        reversed(ATTN_DEPTHS)
    )
    for i, (depth, channels, has_attn) in enumerate(decoder_configs):
        with dot.subgraph(name=f'cluster_up_{i}') as s:
            s.attr(label=f'Up Block {i}')
            
            # Upsample
            if i > 0:
                up_name = f'up_{i}_sample'
                s.node(up_name, 'Upsample\n×2')
                dot.edge(prev_node, up_name)
                prev_node = up_name
            
            # ResBlocks
            for j in range(depth):
                block_name = f'up_{i}_res_{j}'
                label = f'ResBlock\n({channels} ch)'
                if has_attn and j == depth-1:
                    label += '\n+ Attention'
                s.node(block_name, label)
                
                if j == 0:
                    dot.edge(prev_node, block_name)
                else:
                    s.edge(f'up_{i}_res_{j-1}', block_name)
                
                # Skip connections
                if j == 0:
                    skip_from = f'down_{len(DEPTHS)-i-1}_res_{depth-1}'
                    dot.edge(skip_from, block_name, style='dashed')
            
            prev_node = f'up_{i}_res_{depth-1}'
    
    # Output
    dot.node('output', 'Output\n(B, 64, H, W)')
    dot.edge(prev_node, 'output')
    
    # Save visualization
    dot.render('unet_architecture', format='png', cleanup=True)
    print("Model visualization saved as unet_architecture.png")
    
    # Additional text-based architecture summary
    with open("model_summary.txt", "w") as f:
        f.write("UNet Architecture Summary\n")
        f.write("=======================\n\n")
        f.write("Input shape: (B, 64, H, W)\n")
        f.write("Action embedding shape: (B, 16)\n\n")
        f.write("Architecture details:\n")
        f.write(f"- Number of resolution levels: {len(DEPTHS)}\n")
        f.write(f"- Channel dimensions: {CHANNELS}\n")
        attn_levels = [i for i, att in enumerate(ATTN_DEPTHS) if att]
        f.write(f"- Attention at levels: {attn_levels}\n")
        f.write(f"- ResBlocks per level: {DEPTHS}\n\n")
        f.write("Component hierarchy:\n")
        f.write("1. Encoder path (downsampling)\n")
        for i, (d, c) in enumerate(zip(DEPTHS, CHANNELS)):
            f.write(f"   Level {i}: {d} ResBlocks, {c} channels\n")
        f.write("\n2. Middle blocks\n")
        mid_ch = CHANNELS[-1]
        f.write(f"   2 ResBlocks with {mid_ch} channels and attention\n\n")
        f.write("3. Decoder path (upsampling)\n")
        for i, (d, c) in enumerate(zip(reversed(DEPTHS), reversed(CHANNELS))):
            f.write(f"   Level {i}: {d} ResBlocks, {c} channels\n")


if __name__ == "__main__":
    create_model_visualization()
