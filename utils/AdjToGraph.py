from utils.GraphDrawer import DotGraph

def AdjToGraph(Adj, varnames_for_Adj):

    dotGraph = DotGraph()
    for x in varnames_for_Adj:
        if x.startswith("L"):
            dotGraph.addNodeByColor(x, 'red')
        else:
            dotGraph.addNodeByColor(x, 'blue')

    for i in range(len(Adj)):
        for j in range(i+1, len(Adj)):
            
            if (Adj[i,j]==1 and Adj[j,i]==1) or Adj[i,j]==-1 and Adj[j,i]==-1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j])
            elif Adj[i,j]==-1 and Adj[j,i]==1:
                dotGraph.addEdge(varnames_for_Adj[i], varnames_for_Adj[j], type=1)
            elif Adj[j,i]==-1 and Adj[i,j]==1:
                dotGraph.addEdge(varnames_for_Adj[j], varnames_for_Adj[i], type=1)

    return dotGraph