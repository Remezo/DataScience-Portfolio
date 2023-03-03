using Random: shuffle

# PREDICTION & TRAINING

# Sets all activations in the network based on the input_batch using a
# single forward pass over the layers. Nothing returned.
function predict!(model::DenseNetwork, input_batch::AbstractMatrix{<:Real})
    model.input_layer=transpose(input_batch)
    prev_layer=model.input_layer
    for hidden_layer in model.hidden_layers
        
        hidden_layer.A= hidden_layer.activation((hidden_layer.W*prev_layer).+hidden_layer.B)
        prev_layer=hidden_layer.A
    end
    model.output_layer.A=model.output_layer.activation((model.output_layer.W *prev_layer).+model.output_layer.B)   
      
end

# Computes predictions on an entire data set, working in increments of batch_size,
# returning an array of outputs, where each row is the model's prediction on the
# corresponding row of the inputs.
# If batch_size is 0, all predictions are computed as a single batch.
function predict(model::DenseNetwork, inputs::AbstractMatrix{<:Real}; batch_size::Integer=0)
    
    predictions=zeros(size(inputs,1), length(model.output_layer.A))
    if batch_size > 0
        for i in 0:batch_size:size(inputs,1)
        
            if (i+batch_size)>=size(inputs,1)
                finIdx=size(inputs,1)
            else 
                finIdx=i+batch_size  
            end
            input_batch=inputs[i: finIdx, :]
            predict!(model, input_batch)
            predictions[i: finIdx, :]=transpose(model.output_layer.A)  
        end 
    else 
        predict!(model, inputs)
        predictions=transpose(model.output_layer.A)
    
    end
         
    return predictions
end

# Sets all deltas in the network based on batch_targets using a single
# backward pass over the layers. Assumes that predict! was just called
# on the same batch to set the activations. Nothing returned.
function gradient!(model::DenseNetwork, batch_targets::AbstractMatrix{<:Real})
    
    if model.output_layer.activation==softmax
        
        model.output_layer.Δ=model.output_layer.A.-transpose(batch_targets)
        # idx=findmax(batch_targets)[2]
        # model.output_layer.Δ[idx]=max(model.output_layer.A)-1 
    else 
       model.output_layer.Δ=-2 .*(model.output_layer.A.-transpose(batch_targets)).*model.output_layer.derivative.(model.output_layer.A) 
        
    end
    
    prev_layer=model.output_layer
    for (i,hidden_layer) in enumerate(model.hidden_layers) 
        hidden_layer.Δ= transpose(prev_layer.W)*(prev_layer.Δ) .*(hidden_layer.derivative(hidden_layer.A))
        prev_layer=hidden_layer
    end
    
end

# Updates all weights in the network. Assumes that gradient! was just
# called to set the deltas. Nothing returned.
function update!(model::DenseNetwork, learning_rate::Real)
    
    prev_layer=model.output_layer.Δ
    batch_size=size(model.input_layer,1)
    for hidden_layer in hidden_layers
        hidden_layer.W-= (learning_rate/batch_size)*(prev_layer*transpose(hidden_layer.A))
         hidden_layer.B-= (learning_rate/batch_size)*(sum(prev_layer, dims=1))
    end
     
end

# Performs mini-batch stochastic gradient descent training.
# Assumes the loss function based on the output layer's activations:
#     softmax ==> CCE; anything else ==> SSE.
# Records the loss on the entire data set at the end of each epoch
# and returns a vector of losses.
# If verbose is true, then epoch numbers and losses are printed along the way.
function train!(model::DenseNetwork, inputs::AbstractMatrix{<:Real},
                targets::AbstractMatrix{<:Real}, learning_rate::Real,
                epochs::Integer, batch_size::Integer; verbose=false)
    
    for epoch in 1:epochs
        
        loss=0
        
        for i in 1:batch_size:size(inputs,1)
            finIdx=i+batch_size
            predictions=predict(model, inputs, batch_size)
            gradient!(model, targets[i: finIdx, :])
            update(model, learning_rate)
            
            if model.output_layer.activation==softmax
                
            end
            
        end
        println("number of epochs")
    end 
 
    
    
end



# Suggested helper functions; feel free to use different or additional helpers!


# Computes the activations for a hidden or output layer
# and stores them in the layer's A field.
function set_activations(curr_layer::DenseLayer, prev_layer_activations::AbstractMatrix{<:Real})
    error("unimplemented")
end

# Computes the deltas for an output layer that uses softmax
# activation and categorical cross-entropy loss, and stores
# them in the layer's Δ field.
function set_CCE_output_deltas(targets::AbstractMatrix{<:Real}, output_layer::DenseLayer)
    error("unimplemented")
end

# Computes the deltas for any other type of output layer, using the
# layer's derivative-function, and stores them in the layer's Δ field.
# We'll drop the *2 since this can be absorbed into the learning rate.
function set_SSE_output_deltas(targets::AbstractMatrix{<:Real}, output_layer::DenseLayer)
    error("unimplemented")
end

# Computes the deltas for a hidden layer, using the,
# and stores them in the layer's Δ field.
function set_hidden_deltas(curr_layer::DenseLayer, next_layer::DenseLayer)
    error("unimplemented")
end
