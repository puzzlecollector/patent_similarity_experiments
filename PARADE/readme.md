# Experimenting with PARADE 

Just like PARADE, we split up the document into smaller chunks, then obtain representations of each of the chunks via a pre-trained language model. Then we aggregate those chunks using transformer encoder layers. We finally pool the output from the transformer encoder layer to obtain a 768-d vector representation of the entire document (in our case, would be part of the patent document). 
