/*! @file
 * \brief SuperLU utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Oak Ridge National Lab
 * May 12, 2021
 * </pre>
 */
#include <stdio.h>
#include <assert.h>
#include "superlu_ddefs.h"
#if 0
#include "sec_structs.h"
#include "supernodal_etree.h"
#include "load-balance/supernodalForest.h"
#include "p3dcomm.h"
#endif
#include <math.h>

#define INT_T_ALLOC(x)  ((int_t *) SUPERLU_MALLOC ( (x) * sizeof (int_t)))
#define DOUBLE_ALLOC(x)  ((double *) SUPERLU_MALLOC ( (x) * sizeof (double)))


int_t calcTopInfoForest(sForest_t *forest,
                        int_t nsupers, int_t* setree);


sForest_t**  getForests( int_t maxLvl, int_t nsupers, int_t*setree, treeList_t* treeList)
{
	// treePartStrat tps;
	if (getenv("SUPERLU_LBS"))
	{
		if (strcmp(getenv("SUPERLU_LBS"), "ND" ) == 0)
		{
			return getNestDissForests( maxLvl, nsupers, setree, treeList);
		}
		if (strcmp(getenv("SUPERLU_LBS"), "GD" ) == 0)
		{
			return getGreedyLoadBalForests( maxLvl, nsupers, setree, treeList);
		}
	}
	else
	{
		return getGreedyLoadBalForests( maxLvl, nsupers, setree, treeList);
	}
	return 0;
}

double calcNodeListWeight(int_t nnodes, int_t* nodeList, treeList_t* treeList)
{
	double trWeight = 0;

	for (int i = 0; i < nnodes; ++i)
	{
	    trWeight += treeList[nodeList[i]].weight;
	}

	return trWeight;
}

sForest_t**  getNestDissForests( int_t maxLvl, int_t nsupers, int_t*setree, treeList_t* treeList)
{

	int_t numForests = (1 << maxLvl) - 1;

	// allocate space for forests
	sForest_t**  sForests = SUPERLU_MALLOC (numForests * sizeof (sForest_t*));


	int_t* gTreeHeads = getTreeHeads(maxLvl, nsupers, treeList);

	int_t* gNodeCount = calcNumNodes(maxLvl, gTreeHeads, treeList);
	int_t** gNodeLists = getNodeList(maxLvl, setree, gNodeCount,
	                                 gTreeHeads,  treeList);

	SUPERLU_FREE(gTreeHeads); // Sherry added

	for (int i = 0; i < numForests; ++i)
	{
		sForests[i] = NULL;
		if (gNodeCount[i] > 0)
		{
			sForests[i] = SUPERLU_MALLOC (sizeof (sForest_t));
			sForests[i]->nNodes = gNodeCount[i];
			sForests[i]->numTrees = 1;
			sForests[i]->nodeList = gNodeLists[i];
			sForests[i]->weight = calcNodeListWeight(sForests[i]->nNodes, sForests[i]->nodeList, treeList);

			calcTopInfoForest(sForests[i],  nsupers, setree);
		}
	}
	SUPERLU_FREE(gNodeCount);
	SUPERLU_FREE(gNodeLists);

	return sForests;
}

static int_t* sortPtr;

static  int cmpfuncInd (const void * a, const void * b)
{
	return ( sortPtr[*(int_t*)a] - sortPtr[*(int_t*)b] );
}
// doesn't sort A but gives the index of sorted array
int_t* getSortIndex(int_t n, int_t* A)
{
	int_t* idx = INT_T_ALLOC(n);

	for (int i = 0; i < n; ++i)
	{
		/* code */
		idx[i] = i;
	}
	sortPtr = A;

	qsort(idx, n, sizeof(int_t), cmpfuncInd);

	return idx;
}


static double* sortPtrDouble;

static  int cmpfuncIndDouble (const void * a, const void * b)
{
    return ( sortPtrDouble[*(int_t*)a] > sortPtrDouble[*(int_t*)b] );
}
// doesn't sort A but gives the index of sorted array
int_t* getSortIndexDouble(int_t n, double* A)
{
	int_t* idx = INT_T_ALLOC(n);

	for (int i = 0; i < n; ++i)
	{
	    /* code */
	    idx[i] = i;
	}
	sortPtrDouble = A;

	qsort(idx, n, sizeof(int_t), cmpfuncIndDouble);

	return idx;
}

static int cmpfunc(const void * a, const void * b)
{
	return ( *(int_t*)a - * (int_t*)b );
}


int_t* permuteArr(int_t n, int_t* A, int_t* perm)
{
	int_t* permA = INT_T_ALLOC(n);

	for (int i = 0; i < n; ++i)
	{
		/* code */
		permA[i] = A[perm[i]];
	}

	return permA;
}



int_t calcTopInfoForest(sForest_t *forest,
                        int_t nsupers, int_t* setree)
{

	int_t nnodes = forest->nNodes;
	int_t* nodeList = forest->nodeList;

	qsort(nodeList, nnodes, sizeof(int_t), cmpfunc);
	int_t* myIperm = getMyIperm(nnodes, nsupers, nodeList);
	int_t* myTopOrderOld = getMyTopOrder(nnodes, nodeList, myIperm, setree );
	int_t* myTopSortIdx = getSortIndex(nnodes, myTopOrderOld);
	int_t* nodeListNew = permuteArr(nnodes, nodeList, myTopSortIdx);
	int_t* myTopOrder = permuteArr(nnodes, myTopOrderOld, myTopSortIdx);

	SUPERLU_FREE(nodeList);
	SUPERLU_FREE(myTopSortIdx);
	SUPERLU_FREE(myIperm);
	SUPERLU_FREE(myTopOrderOld);
	myIperm = getMyIperm(nnodes, nsupers, nodeListNew);

	treeTopoInfo_t ttI;
	ttI.myIperm = myIperm;
	ttI.numLvl = myTopOrder[nnodes - 1] + 1;
	ttI.eTreeTopLims = getMyEtLims(nnodes, myTopOrder);

	forest->nodeList = nodeListNew;
	forest->topoInfo = ttI;

	SUPERLU_FREE(myTopOrder); // sherry added

	return 0;
}

// #pragma optimize ("", off)

double* getTreeWeights(int_t numTrees, int_t* gNodeCount, int_t** gNodeLists, treeList_t* treeList)
{
	double* gTreeWeights = DOUBLE_ALLOC(numTrees);

	// initialize with weight with whole subtree weights
	for (int_t i = 0; i < numTrees; ++i)
	{
	    gTreeWeights[i] = calcNodeListWeight(gNodeCount[i], gNodeLists[i], treeList);
	}

	return gTreeWeights;

}

int_t*  getNodeCountsFr(int_t maxLvl, sForest_t**  sForests)
{
    int_t numForests = (1 << maxLvl) - 1;
    int_t* gNodeCount = INT_T_ALLOC (numForests);

    for (int i = 0; i < numForests; ++i)
	{
	    /* code */
	    if (sForests[i])
		{gNodeCount[i] = sForests[i]->nNodes;}
	    else
		{
		    gNodeCount[i] = 0;
		}
	}
	return gNodeCount;
}

int_t** getNodeListFr(int_t maxLvl, sForest_t**  sForests)
{
	int_t numForests = (1 << maxLvl) - 1;
	int_t** gNodeLists =  (int_t**) SUPERLU_MALLOC(numForests * sizeof(int_t*));

	for (int i = 0; i < numForests; ++i)
	{
		/* code */
		if (sForests[i])
		{
			gNodeLists[i] = sForests[i]->nodeList;
		}
		else
		{
			gNodeLists[i] = NULL;
		}
	}

	return  gNodeLists;
}

int_t* getNodeToForstMap(int_t nsupers, sForest_t**  sForests, gridinfo3d_t* grid3d)
{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t numForests = (1 << maxLvl) - 1;
	int_t* gNodeToForstMap = INT_T_ALLOC (nsupers);
	
	for (int i = 0; i < numForests; ++i)
	{
		/* code */
		if (sForests[i])
		{	int_t nnodes = sForests[i]->nNodes;
			int_t* nodeList = sForests[i]->nodeList;
			for(int_t node = 0; node<nnodes; node++)
			{
				gNodeToForstMap[nodeList[node]]	=i;
			}
		}
	}

	return gNodeToForstMap;

}

int_t* getMyNodeCountsFr(int_t maxLvl, int_t* myTreeIdxs, sForest_t**  sForests)
{
	int_t* myNodeCount = INT_T_ALLOC(maxLvl);
	for (int i = 0; i < maxLvl; ++i)
	{
		myNodeCount[i] = 0;
		if (sForests[myTreeIdxs[i]])
			myNodeCount[i] = sForests[myTreeIdxs[i]]->nNodes;
	}

	return myNodeCount;
}


int_t** getTreePermFr( int_t* myTreeIdxs,
                       sForest_t**  sForests, gridinfo3d_t* grid3d)
{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;

	int_t** treePerm = (int_t** ) SUPERLU_MALLOC(sizeof(int_t*)*maxLvl);
	for (int_t lvl = 0; lvl < maxLvl; lvl++)
	{
		treePerm[lvl] = NULL;
		if (sForests[myTreeIdxs[lvl]])
			treePerm[lvl] = sForests[myTreeIdxs[lvl]]->nodeList;
	}
	return treePerm;
}

int* getIsNodeInMyGrid(int_t nsupers, int_t maxLvl, int_t* myNodeCount, int_t** treePerm)
{
    int* isNodeInMyGrid = SUPERLU_MALLOC(nsupers * sizeof(int));

    for(int i=0; i<nsupers; i++) isNodeInMyGrid[i] =0;

    for (int i = 0; i < maxLvl; ++i)
    {
	for(int node = 0; node< myNodeCount[i]; node++ )
	{
	    isNodeInMyGrid[treePerm[i][node]]=1;
	}
    }

    return isNodeInMyGrid;
}

double pearsonCoeff(int_t numForests, double* frCost, double* frWeight)
{
	if (numForests == 1)
	{
		return 1.0;
	}
	double meanCost = 0;
	double meanWeight = 0;
	for (int i = 0; i < numForests; ++i)
	{
		meanWeight += frWeight[i] / numForests;
		meanCost += frCost[i] / numForests;
	}

	double stdCost = 0;
	double stdWeight = 0;
	double covarCostWeight = 0;
	for (int i = 0; i < numForests; ++i)
	{
		/* code */
		stdCost += (frCost[i] - meanCost) * (frCost[i] - meanCost);
		stdWeight += (frWeight[i] - meanWeight) * (frWeight[i] - meanWeight);
		covarCostWeight += (frCost[i] - meanCost) * (frWeight[i] - meanWeight);
	}

	return covarCostWeight / sqrt(stdCost * stdWeight);

}
void printGantt(int root, int numForests, char* nodename, double scale, double* gFrstCostAcc, double* crPathCost);

void printForestWeightCost(sForest_t**  sForests, SCT_t* SCT, gridinfo3d_t* grid3d)
{
	gridinfo_t* grid = &(grid3d->grid2d);
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t numForests = (1 << maxLvl) - 1;
	double* gFrstCost = DOUBLE_ALLOC(numForests);
	double* gFrstCostAcc = DOUBLE_ALLOC(numForests);
	double* gFrstWt = DOUBLE_ALLOC(numForests);

	for (int i = 0; i < numForests; ++i)
	{
		gFrstCost[i] = 0;
		gFrstWt[i] = 0;
		if (sForests[i])
		{
			gFrstCost[i] = sForests[i]->cost;
			gFrstWt[i] = sForests[i]->weight;
		}
	}

	// reduce forest costs from all the grid;
	MPI_Reduce(gFrstCost, gFrstCostAcc, numForests, MPI_DOUBLE, MPI_SUM, 0, grid3d->zscp.comm);

	if (!grid3d->zscp.Iam  && !grid->iam)
	{
	    printf("|Forest | weight | cost | weight/Cost | \n");
	    for (int i = 0; i < numForests; ++i)
		{
		    /* code */
		    double wt, ct;
		    wt = 0.0;
		    ct = 0.0;
		    if (sForests[i])
			{
			    wt = sForests[i]->weight;
			}
		    printf("|%d   | %.2e   | %.2e   | %.2e  |\n", i, wt, gFrstCostAcc[i], 1e-9 * wt / gFrstCostAcc[i] );

		}

		double* crPathCost = DOUBLE_ALLOC(numForests);
		double* crPathWeight = DOUBLE_ALLOC(numForests);
		// print the critcal path
		for (int i = numForests - 1; i > -1 ; --i)
		{
		    crPathCost[i] = gFrstCostAcc[i];
		    crPathWeight[i] = gFrstWt[i];

		    if (2 * i + 1 < numForests)
			{

			    if (crPathCost[2 * i + 1] > crPathCost[2 * i + 2])
				{
				    /* code */
				    crPathCost[i] += crPathCost[2 * i + 1];
				    crPathWeight[i] += crPathWeight[2 * i + 1];
				}
			    else
				{
				    crPathCost[i] += crPathCost[2 * i + 2];
				    crPathWeight[i] += crPathWeight[2 * i + 2];
				}
			}
		}


		printf("|CritcalPath   | %.2e   | %.2e   | %.2e  |\n", crPathWeight[0], crPathCost[0], 1e-9 * crPathWeight[0] / crPathCost[0] );

		double prsnCoeff = pearsonCoeff(numForests, gFrstCost, gFrstWt);
		printf("|Pearsoncoefficient |  %.3f |\n", prsnCoeff);

		printf("\n~~~mermaid \n");
		printf("\ngantt \n \
       \t\t dateFormat  mm-ss \n\
       \t\t title TreeCost and Time Gantt Chart\n\n\n" );
		printf("\t Section Time\n");
		printGantt(0,  numForests, "Time", 1.0 , gFrstCostAcc, crPathCost);
		printf("\t Section Weight\n");
		printGantt(0,  numForests, "weight", crPathCost[0]/crPathWeight[0] , gFrstWt, crPathWeight);

		printf("~~~\n\n\n");
		SUPERLU_FREE(crPathCost);
		SUPERLU_FREE(crPathWeight);

	}

	SUPERLU_FREE( gFrstCost);
	SUPERLU_FREE( gFrstCostAcc);
	SUPERLU_FREE( gFrstWt);
}


void printGantt(int root, int numForests, char* nodename, double scale, double* gFrstCostAcc, double* crPathCost)
{


	if (2*root+1>=numForests)
	{
		/* if there are no more childrens*/
		printf("\t tree-%d \t:%s-%d, 0d, %.0fd \n", root,nodename, root,  100*scale*gFrstCostAcc[root]  );
	}
	else
	{
	  printGantt(2*root+1,  numForests, nodename,  scale,  gFrstCostAcc, crPathCost);
	  int depTree =crPathCost[2*root+1]> crPathCost[2*root+2]? 2*root+1:2*root+2;
	  printf("\t tree-%d  %.2g \t:%s-%d, after %s-%d, %.0fd \n", root,100*scale*crPathCost[root], nodename, root, nodename, depTree, 100*scale*gFrstCostAcc[root]  );
	  printGantt(2*root+2,  numForests,  nodename, scale,  gFrstCostAcc, crPathCost);	
	}
}

#define ABS(a) ((a)<0?-(a):a)
double getLoadImbalance(int_t nTrees,
                        int_t * treeIndx,				// index of tree in gtrees
                        double * gTreeWeights)
{

	if (nTrees < 1)
	{
		/* code */
		return 0;
	}
	double w1 = 0;
	double w2 = 0;

	int_t* wSortIdx = getSortIndexDouble(nTrees, gTreeWeights);
	// can not change weight array
	w1 = gTreeWeights[wSortIdx[nTrees - 1]];


	for (int i = nTrees - 2 ; i > -1; --i)
	{
		/* code */
		if (w1 > w2)
		{
			/* code */
			w2 += gTreeWeights[wSortIdx[i]];

		}
		else
		{
			w1 += gTreeWeights[wSortIdx[i]];

		}
	}

	SUPERLU_FREE(wSortIdx);
	return ABS(w2 - w1) / (w2 + w1);
	// return trPart;

}


// maximum allowed imbalance
#define ACCEPTABLE_TREE_IMBALANCE 0.2


// r forest contains a list of tree heads
// each treehead is an entire subtree (all level beloe)
// #define MAX_TREE_ALLOWED 1024
// #define MAX_TREE_ALLOWED 2048
#define NUM_TREE_LOWERB 32

typedef struct
{
	int_t ntrees;
	int_t* treeHeads;
} rForest_t;

typedef struct
{
	sForest_t* Ans;
	rForest_t* S[2];
} forestPartition_t;

void freeRforest(rForest_t* rforest)
{
	SUPERLU_FREE(rforest->treeHeads);
}


sForest_t*  createForestNew(int_t numTrees, int_t nsupers, int_t * nodeCounts,  int_t** NodeLists, int_t * setree, treeList_t* treeList)
{
	if (numTrees == 0) return NULL;

	sForest_t* forest = SUPERLU_MALLOC(sizeof(sForest_t));
	forest->numTrees = numTrees;

	double frWeight = 0;
	int_t nodecount = 0;
	for (int_t i = 0; i < numTrees; ++i)
	{
	    nodecount += nodeCounts[i];
	    frWeight += calcNodeListWeight(nodeCounts[i], NodeLists[i], treeList);
	}

	forest->nNodes = nodecount;
	forest->weight = frWeight;

	int_t* nodeList = INT_T_ALLOC(forest->nNodes);

	int_t ptr = 0;
	for (int_t i = 0; i < numTrees; ++i)
	{
	    for (int_t j = 0; j < nodeCounts[i]; ++j)
		{
		    /* copy the loop */
		    nodeList[ptr] = NodeLists[i][j];
		    ptr++;
		}
	}

	forest->nodeList = nodeList;
	forest->cost = 0.0;


	// using the nodelist create factorization ordering
	calcTopInfoForest(forest, nsupers, setree);

	return forest;
}

void oneLeveltreeFrPartition( int_t nTrees, int_t * trCount, int_t** trList,
                              int_t * treeSet,
                              double * sWeightArr)
{
	if (nTrees < 1)
	{
		/* code */
		trCount[0] = 0;
		trCount[1] = 0;
		return;
	}
	double w1 = 0;
	double w2 = 0;

	int_t* wSortIdx = getSortIndexDouble(nTrees, sWeightArr);
	// treeIndx= permuteArr(nTrees, treeIndx, wSortIdx);

	int_t S1ptr = 0;
	int_t S2ptr = 0;

	// can not change weight array
	w1 = sWeightArr[wSortIdx[nTrees - 1]];
	trList[0][S1ptr++] = treeSet[wSortIdx[nTrees - 1]];

	for (int i = nTrees - 2 ; i > -1; --i)
	{
		/* code */
		if (w1 > w2)
		{
			/* code */
			w2 += sWeightArr[wSortIdx[i]];
			trList[1][S2ptr++] = treeSet[wSortIdx[i]];
		}
		else
		{
			w1 += sWeightArr[wSortIdx[i]];
			trList[0][S1ptr++] = treeSet[wSortIdx[i]];
		}
	}

	trCount[0] = S1ptr;
	trCount[1] = S2ptr;

	SUPERLU_FREE(wSortIdx);

} /* oneLeveltreeFrPartition */

void resizeArr(void** A, int oldSize, int newSize, size_t typeSize)
{
	assert(newSize>oldSize);
	if(newSize==oldSize) return; 

	void* newPtr = SUPERLU_MALLOC(newSize * typeSize);

	// copy *A to new ptr upto oldSize
	memcpy(newPtr, *A, oldSize * typeSize);
	// free the memory

	SUPERLU_FREE(*A);

	*A = newPtr;

	return; 

}
forestPartition_t iterativeFrPartitioning(rForest_t* rforest, int_t nsupers, int_t * setree, treeList_t* treeList)
{

    int_t nTreeSet = rforest->ntrees;
    int_t* treeHeads =  rforest->treeHeads;

    int_t nAnc = 0;

	int treeArrSize = SUPERLU_MAX( 2*nTreeSet, NUM_TREE_LOWERB) ;
    int_t* ancTreeCount = intMalloc_dist(treeArrSize);
    int_t** ancNodeLists = SUPERLU_MALLOC(treeArrSize * sizeof(int_t*));

    double * weightArr = doubleMalloc_dist(treeArrSize);
    int_t* treeSet = intMalloc_dist(treeArrSize);
	

	for (int i = 0; i < nTreeSet; ++i)
	{
		treeSet[i] = treeHeads[i];
		weightArr[i] = treeList[treeHeads[i]].iWeight;
	}

	while (getLoadImbalance(nTreeSet, treeSet, weightArr) > ACCEPTABLE_TREE_IMBALANCE )
	{
		// get index of maximum weight subtree
		int_t idx = 0;
		for (int i = 0; i < nTreeSet; ++i)
		{
			/* code */
			if (treeList[treeSet[i]].iWeight > treeList[treeSet[idx]].iWeight)
			{
				/* code */
				idx = i;
			}
		}


		int_t MaxTree = treeSet[idx];
		int_t numSubtrees;
		int_t*  sroots = getSubTreeRoots(MaxTree, &numSubtrees, treeList);
		if (numSubtrees==0)
		{
			break;
		}

		ancTreeCount[nAnc] = getCommonAncsCount(MaxTree, treeList);
		//int_t * alist = INT_T_ALLOC (ancTreeCount[nAnc]);
		int_t * alist = intMalloc_dist(ancTreeCount[nAnc]);
		getCommonAncestorList(MaxTree, alist, setree, treeList);
		ancNodeLists[nAnc] = alist;
		nAnc++;

		// treeSet[idx] is removed and numsubtrees are added
		int newNumTrees= nTreeSet - 1 + numSubtrees;

		if(newNumTrees>treeArrSize)
		{
			// double the array size 
			// resizeArr(void** A, int oldSize, int newSize, size_t typeSize);
			resizeArr( (void**) &ancTreeCount, treeArrSize, 2*newNumTrees, sizeof(int_t));
			resizeArr( (void**) &ancNodeLists, treeArrSize, 2*newNumTrees, sizeof(int_t*));
			resizeArr( (void**) &weightArr, treeArrSize, 2*newNumTrees, sizeof(double));
			resizeArr( (void**) &treeSet, treeArrSize, 2*newNumTrees, sizeof(int_t));
			treeArrSize = 2*newNumTrees; 
		}

		//TODO: fix it for multiple children 
		treeSet[idx] = treeSet[nTreeSet - 1];
		weightArr[idx] = treeList[treeSet[idx]].iWeight;

		#if(1)
		for(int j=0; j<numSubtrees; j++)
		{
			treeSet[nTreeSet - 1+j] = sroots[j];
			weightArr[nTreeSet - 1+j] = treeList[sroots[j]].iWeight;		
		}
		nTreeSet = newNumTrees;
		#else 
		treeSet[nTreeSet - 1] = sroots[0];
		weightArr[nTreeSet - 1] = treeList[treeSet[nTreeSet - 1]].iWeight;
		treeSet[nTreeSet] = sroots[1];
		weightArr[nTreeSet] = treeList[treeSet[nTreeSet]].iWeight;
		nTreeSet += 1;
		#endif 
		SUPERLU_FREE(sroots);

		//TODO: incorrect fix it; 
		// if (nTreeSet == MAX_TREE_ALLOWED)
		// {
		// 	break;
		// }
	}

	// Create the Ancestor forest
	sForest_t* aforest = createForestNew(nAnc, nsupers, ancTreeCount, ancNodeLists, setree, treeList);

	// create the weight array;
	//double* sWeightArr = DOUBLE_ALLOC(nTreeSet);
	double* sWeightArr = doubleMalloc_dist(nTreeSet); // Sherry fix
	for (int i = 0; i < nTreeSet ; ++i)
		sWeightArr[i] = treeList[treeSet[i]].iWeight;

	int_t trCount[2] = {0, 0};
	int_t* trList[2];
#if 0
	trList[0] = INT_T_ALLOC(nTreeSet);
	trList[1] = INT_T_ALLOC(nTreeSet);
#else  // Sherry fix
	trList[0] = intMalloc_dist(nTreeSet);
	trList[1] = intMalloc_dist(nTreeSet);
#endif

	oneLeveltreeFrPartition( nTreeSet, trCount, trList,
	                         treeSet,
	                         sWeightArr);

	rForest_t *rforestS1, *rforestS2;
#if 0
	rforestS1 = SUPERLU_MALLOC(sizeof(rforest));
	rforestS2 = SUPERLU_MALLOC(sizeof(rforest));
#else
	rforestS1 = (rForest_t *) SUPERLU_MALLOC(sizeof(rForest_t));  // Sherry fix
	rforestS2 = (rForest_t *) SUPERLU_MALLOC(sizeof(rForest_t));
#endif

	rforestS1->ntrees = trCount[0];
	rforestS1->treeHeads = trList[0];

	rforestS2->ntrees = trCount[1];
	rforestS2->treeHeads = trList[1];

	forestPartition_t frPr_t;
	frPr_t.Ans 	= aforest;
	frPr_t.S[0]     = rforestS1;
	frPr_t.S[1]	= rforestS2;

	SUPERLU_FREE(weightArr);
	SUPERLU_FREE(treeSet);
	SUPERLU_FREE(sWeightArr);

	// free stuff
	// 	int_t* ancTreeCount = INT_T_ALLOC(MAX_TREE_ALLOWED);
	// int_t** ancNodeLists = SUPERLU_MALLOC(MAX_TREE_ALLOWED * sizeof(int_t*));

	for (int i = 0; i < nAnc ; ++i)
	{
		/* code */
		SUPERLU_FREE(ancNodeLists[i]);
	}

	SUPERLU_FREE(ancTreeCount);
	SUPERLU_FREE(ancNodeLists);

	return frPr_t;
} /* iterativeFrPartitioning */


/* Create a single sforest */
sForest_t* r2sForest(rForest_t* rforest, int_t nsupers, int_t * setree, treeList_t* treeList)
{
	int_t nTree = rforest->ntrees;

	// quick return
	if (nTree < 1) return NULL;

	int_t* treeHeads =  rforest->treeHeads;
	int_t* nodeCounts = INT_T_ALLOC(nTree);
	int_t** NodeLists = SUPERLU_MALLOC(nTree * sizeof(int_t*));

	for (int i = 0; i < nTree; ++i)
	{
	    /* code */
	    nodeCounts[i] = treeList[treeHeads[i]].numDescendents;
	    NodeLists[i] = INT_T_ALLOC(nodeCounts[i]);
	    getDescendList(treeHeads[i], NodeLists[i], treeList);
	}


	sForest_t* sforest =  createForestNew(nTree, nsupers,  nodeCounts,  NodeLists, setree, treeList);

	for (int i = 0; i < nTree; ++i)
	{
		/* code */
		SUPERLU_FREE(NodeLists[i]);
	}

	SUPERLU_FREE(NodeLists);
	SUPERLU_FREE(nodeCounts);

	return sforest;
} /* r2sForest */


sForest_t**  getGreedyLoadBalForests( int_t maxLvl, int_t nsupers, int_t * setree, treeList_t* treeList)
{

	// assert(maxLvl == 2);
	int_t numForests = (1 << maxLvl) - 1;
	sForest_t**  sForests = (sForest_t** ) SUPERLU_MALLOC (numForests * sizeof (sForest_t*));

	int_t numRForests = SUPERLU_MAX( (1 << (maxLvl - 1)) - 1, 1) ;
	rForest_t*  rForests = SUPERLU_MALLOC (numRForests * sizeof (rForest_t));

	// intialize rfortes[0]
	int_t nRootTrees = 0;

	for (int i = 0; i < nsupers; ++i)
	{
		/* code */
		if (setree[i] == nsupers) nRootTrees++;

	}

	rForests[0].ntrees = nRootTrees;
	rForests[0].treeHeads = INT_T_ALLOC(nRootTrees);

	nRootTrees = 0;
	for (int i = 0; i < nsupers; ++i)
	{
		/* code */
		if (setree[i] == nsupers)
		{
			rForests[0].treeHeads[nRootTrees] = i;
			nRootTrees++;
		}

	}

	if (maxLvl == 1)
	{
		/* code */
		sForests[0] = r2sForest(&rForests[0], nsupers, setree, treeList);

		freeRforest(&rForests[0]);  // sherry added
		SUPERLU_FREE(rForests);
		return sForests;
	}

	// now loop over level
	for (int_t lvl = 0; lvl < maxLvl - 1; ++lvl)
	{
		/* loop over all r forest in this level */
		int_t lvlSt = (1 << lvl) - 1;
		int_t lvlEnd = (1 << (lvl + 1)) - 1;

		for (int_t tr = lvlSt; tr < lvlEnd; ++tr)
		{
		    /* code */
		    forestPartition_t frPr_t = iterativeFrPartitioning(&rForests[tr], nsupers, setree, treeList);
		    sForests[tr] = frPr_t.Ans;

		    if (lvl == maxLvl - 2) {
			/* code */
			sForests[2 * tr + 1] = r2sForest(frPr_t.S[0], nsupers, setree, treeList);
			sForests[2 * tr + 2] = r2sForest(frPr_t.S[1], nsupers, setree, treeList);
			freeRforest(frPr_t.S[0]); // Sherry added
			freeRforest(frPr_t.S[1]);
#if 0
			SUPERLU_FREE(frPr_t.S[0]); // Sherry added
			SUPERLU_FREE(frPr_t.S[1]);
#endif
		    } else {
			rForests[2 * tr + 1] = *(frPr_t.S[0]);
			rForests[2 * tr + 2] = *(frPr_t.S[1]);
			
		    }
		    SUPERLU_FREE(frPr_t.S[0]); // Sherry added
		    SUPERLU_FREE(frPr_t.S[1]);
		}

	}

	for (int i = 0; i < numRForests; ++i)
	{
	    /* code */
	    freeRforest(&rForests[i]);  // Sherry added
	}

	SUPERLU_FREE(rForests);  // Sherry added

	return sForests;

} /* getGreedyLoadBalForests */

// balanced forests at one level
sForest_t**  getOneLevelBalForests( int_t maxLvl, int_t nsupers, int_t * setree, treeList_t* treeList)
{

	// assert(maxLvl == 2);
	int_t numForests = (1 << maxLvl) - 1;
	sForest_t**  sForests = (sForest_t** ) SUPERLU_MALLOC (numForests * sizeof (sForest_t*));

	int_t numRForests = SUPERLU_MAX( (1 << (maxLvl - 1)) - 1, 1) ;
	rForest_t*  rForests = SUPERLU_MALLOC (numRForests * sizeof (rForest_t));

	// intialize rfortes[0]
	int_t nRootTrees = 0;

	for (int i = 0; i < nsupers; ++i)
	{
		/* code */
		if (setree[i] == nsupers)
		{
			nRootTrees += 2;
		}

	}

	rForests[0].ntrees = nRootTrees;
	rForests[0].treeHeads = INT_T_ALLOC(nRootTrees);

	nRootTrees = 0;
	for (int i = 0; i < nsupers; ++i)
	{
		/* code */
		if (setree[i] == nsupers)
		{
			rForests[0].treeHeads[nRootTrees] = i;
			nRootTrees++;
		}
	}

	if (maxLvl == 1)
	{
		/* code */
		sForests[0] = r2sForest(&rForests[0], nsupers, setree, treeList);
		return sForests;
	}

	// now loop over level
	for (int_t lvl = 0; lvl < maxLvl - 1; ++lvl)
	{
		/* loop over all r forest in this level */
		int_t lvlSt = (1 << lvl) - 1;
		int_t lvlEnd = (1 << (lvl + 1)) - 1;

		for (int_t tr = lvlSt; tr < lvlEnd; ++tr)
		{
			/* code */
			forestPartition_t frPr_t = iterativeFrPartitioning(&rForests[tr], nsupers, setree, treeList);
			sForests[tr] = frPr_t.Ans;

			if (lvl == maxLvl - 2)
			{
				/* code */
				sForests[2 * tr + 1] = r2sForest(frPr_t.S[0], nsupers, setree, treeList);
				sForests[2 * tr + 2] = r2sForest(frPr_t.S[1], nsupers, setree, treeList);
			}
			else
			{
				rForests[2 * tr + 1] = *(frPr_t.S[0]);
				rForests[2 * tr + 2] = *(frPr_t.S[1]);
			}

		}

	}

	for (int i = 0; i < numRForests; ++i)
	{
		/* code */
		freeRforest(&rForests[i]);
	}

	SUPERLU_FREE(rForests);



	return sForests;

}


int* getBrecvTree(int_t nlb, sForest_t* sforest,  int* bmod, gridinfo_t * grid)
{
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return NULL;
    int_t *nodeList = sforest->nodeList ;

    // int_t Pr = grid->nprow;
    // int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    superlu_scope_t *scp = &grid->rscp;



    int* mod_bit = SUPERLU_MALLOC(sizeof(int) * nlb);
    for (int_t k = 0; k < nlb; ++k)
        mod_bit[k] = 0;

    int* brecv = SUPERLU_MALLOC(sizeof(int) * nlb);


    for (int_t k0 = 0; k0 < nnodes ; ++k0)
    {
        /* code */
        int_t k = nodeList[k0];
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* local block number */
            int_t kcol = PCOL (k, grid);  /* root process in this row scope */
            if (mycol != kcol )
                mod_bit[lk] = 1;    /* Contribution from off-diagonal */
        }
    }

    /* Every process receives the count, but it is only useful on the
       diagonal processes.  */
    MPI_Allreduce (mod_bit, brecv, nlb, MPI_INT, MPI_SUM, scp->comm);

    SUPERLU_FREE(mod_bit);
    return brecv;
}



int* getBrecvTree_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, int* bmod, gridinfo_t * grid)
{
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    superlu_scope_t *scp = &grid->rscp;



    int* mod_bit = SUPERLU_MALLOC(sizeof(int) * nlb);
    for (int_t k = 0; k < nlb; ++k)
        mod_bit[k] = 0;

    int* brecv = SUPERLU_MALLOC(sizeof(int) * nlb);


	for (int_t k = 0; k < nsupers; ++k)
	{
		if(supernodeMask[k]>0)
        {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* local block number */
            int_t kcol = PCOL (k, grid);  /* root process in this row scope */
            if (mycol != kcol && bmod[lk])
                mod_bit[lk] = 1;    /* Contribution from off-diagonal */
        }
        }
    }

    /* Every process receives the count, but it is only useful on the
       diagonal processes.  */
    MPI_Allreduce (mod_bit, brecv, nlb, MPI_INT, MPI_SUM, scp->comm);

    SUPERLU_FREE(mod_bit);
    return brecv;
}

int getNrootUsolveTree(int_t* nbrecvmod, sForest_t* sforest, int* brecv, int* bmod, gridinfo_t * grid)
{
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 0;
    int_t *nodeList = sforest->nodeList ;

    // int_t Pr = grid->nprow;
    // int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int nroot = 0;
    for (int_t k0 = 0; k0 < nnodes ; ++k0)
    {
        /* code */
        int_t k = nodeList[k0];
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* local block number */
            int_t kcol = PCOL (k, grid);  /* root process in this row scope. */
            if (mycol == kcol)
            {
                /* diagonal process */
                *nbrecvmod += brecv[lk];
                if (!brecv[lk] && !bmod[lk])
                    ++nroot;

            }
        }
    }

    return nroot;
}


int getNbrecvX(sForest_t* sforest, int_t* Urbs, gridinfo_t * grid)
{
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return 0;
    int_t *nodeList = sforest->nodeList ;

    // int_t Pr = grid->nprow;
    // int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int nbrecvx = 0;
    for (int_t k0 = 0; k0 < nnodes ; ++k0)
    {
        /* code */
        int_t k = nodeList[k0];
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (mycol == kcol && myrow != krow)
        {
            /* code */
            int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
            int_t nub = Urbs[lk];      /* Number of U blocks in block column lk */
            if (nub > 0)
                nbrecvx++;
        }
    }

    return nbrecvx;
}

int getNbrecvX_newsolve(int_t nsupers, int* supernodeMask, int_t* Urbs, Ucb_indptr_t **Ucb_indptr, gridinfo_t * grid)
{

    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int nbrecvx = 0;
    int_t ik;

	for (int_t k = 0; k < nsupers; ++k)
	{
        if(supernodeMask[k]>0)
        {
        int_t krow = PROW (k, grid);
        int_t kcol = PCOL (k, grid);
        if (mycol == kcol && myrow != krow)
        {
            /* code */
            int_t lk = LBj( k, grid ); /* Local block number, column-wise. */
            int_t nub = Urbs[lk];      /* Number of U blocks in block column lk */
            int_t flag=0;
            for (int_t ub = 0; ub < nub; ++ub) {
                ik = Ucb_indptr[lk][ub].lbnum; /* Local block number, row-wise. */
                int_t gik = ik * grid->nprow + myrow;/* Global block number, row-wise. */
                if(supernodeMask[gik]>0)
                    flag=1;
            }
            if(flag==1)
                nbrecvx++;
        }
        }
    }

    return nbrecvx;
}


int getNrootUsolveTree_newsolve(int_t* nbrecvmod, int_t nsupers, int* supernodeMask, int* brecv, int* bmod, gridinfo_t * grid)
{

    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    int_t mycol = MYCOL (iam, grid);
    int nroot = 0;
	for (int_t k = 0; k < nsupers; ++k)
	{
        if(supernodeMask[k]>0)
        {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid); /* local block number */
            int_t kcol = PCOL (k, grid);  /* root process in this row scope. */
            if (mycol == kcol)
            {
                /* diagonal process */
                *nbrecvmod += brecv[lk];
                if (!brecv[lk] && !bmod[lk])
                    ++nroot;

            }
        }
        }
    }

    return nroot;
}


int_t getNfrecvmodLeaf(int* nleaf, sForest_t* sforest, int* frecv, int* fmod, gridinfo_t * grid)
{
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	int_t nnodes =   sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;
	int_t nfrecvmod = 0;
	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow)
		{
			int_t lk = LBi (k, grid); /* local block number */
			int_t kcol = PCOL (k, grid);
			if (mycol == kcol)
			{
				/* diagonal process */
				nfrecvmod += frecv[lk];
				if (!frecv[lk] && !fmod[lk])
					++(*nleaf);
			}
		}
	}
	return nfrecvmod;
}

int_t getNfrecvmod_newsolve(int* nleaf, int_t nsupers, int* supernodeMask, int* frecv, int* fmod, gridinfo_t * grid)
{
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	int_t nfrecvmod = 0;
	for (int_t k = 0; k < nsupers; ++k)
	{
        // printf("anni k %5d supernodeMask[k] %5d\n",k,supernodeMask[k]);
        if(supernodeMask[k]>0)
        {
            int_t krow = PROW (k, grid);
            int_t kcol = PCOL (k, grid);

            if (myrow == krow)
            {
                int_t lk = LBi (k, grid); /* local block number */
                int_t kcol = PCOL (k, grid);
                if (mycol == kcol)
                {
                    /* diagonal process */
                    nfrecvmod += frecv[lk];
                    if (!frecv[lk] && !fmod[lk])
                        ++(*nleaf);
                }
            }
        }
	}
	return nfrecvmod;
}


int_t zAllocBcast(int_t size, void** ptr, gridinfo3d_t* grid3d)
{

	if (size < 1) return 0;
	if (grid3d->zscp.Iam)
	{
		*ptr = NULL;
		*ptr = SUPERLU_MALLOC(size);
	}
	MPI_Bcast(*ptr, size, MPI_BYTE, 0, grid3d->zscp.comm);

	return 0;
}



int_t zAllocBcast_gridID(int_t size, void** ptr, int_t gridID, gridinfo3d_t* grid3d)
{

	if (size < 1) return 0;
	if (grid3d->zscp.Iam != gridID)
	{
		*ptr = NULL;
		*ptr = SUPERLU_MALLOC(size);
	}
	MPI_Bcast(*ptr, size, MPI_BYTE, gridID, grid3d->zscp.comm);

	return 0;
}


int* getfrecv_newsolve(int_t nsupers, int* supernodeMask, int_t nlb, int* fmod,
                     int *mod_bit, gridinfo_t * grid)
{

	int* frecv;
	if (!(frecv = int32Malloc_dist (nlb)))
		ABORT ("Malloc fails for frecv[].");
	superlu_scope_t *scp = &grid->rscp;
	for (int_t k = 0; k < nlb; ++k)
		mod_bit[k] = 0;
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	for (int_t k = 0; k < nsupers; ++k)
	{
		if(supernodeMask[k]>0)
        {
            int_t krow = PROW (k, grid);
            int_t kcol = PCOL (k, grid);

            if (myrow == krow)
            {
                int_t lk = LBi (k, grid); /* local block number */
                int_t kcol = PCOL (k, grid);
                if (mycol != kcol && fmod[lk])
                    mod_bit[lk] = 1;    /* contribution from off-diagonal */
            }
        }
	}
	/* Every process receives the count, but it is only useful on the
	   diagonal processes.  */
	MPI_Allreduce (mod_bit, frecv, nlb, MPI_INT, MPI_SUM, scp->comm);

	// for (int_t i = 0; i < nlb; ++i){
    //     printf("id %5d i %5d frecv %5d\n",iam,i,frecv[i]);
    // }

	return frecv;
}


int* getfrecvLeaf( sForest_t* sforest, int_t nlb, int* fmod,
                     int *mod_bit, gridinfo_t * grid)
{

	int* frecv;
	if (!(frecv = int32Malloc_dist (nlb)))
		ABORT ("Malloc fails for frecv[].");
	superlu_scope_t *scp = &grid->rscp;
	for (int_t k = 0; k < nlb; ++k)
		mod_bit[k] = 0;
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	int_t nnodes =   sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;
	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (myrow == krow)
		{
			int_t lk = LBi (k, grid); /* local block number */
			int_t kcol = PCOL (k, grid);
			if (mycol != kcol && fmod[lk])
				mod_bit[lk] = 1;    /* contribution from off-diagonal */
		}
	}
	/* Every process receives the count, but it is only useful on the
	   diagonal processes.  */
	MPI_Allreduce (mod_bit, frecv, nlb, MPI_INT, MPI_SUM, scp->comm);


	return frecv;
}

int getNfrecvx_newsolve(int_t nsupers, int* supernodeMask, int_t** Lrowind_bc_ptr, int_t** Lindval_loc_bc_ptr, gridinfo_t * grid)
{
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	// int_t** Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
	int nfrecvx = 0;
    int_t nb, idx_n, idx_i, lb, lptr1_tmp, ik;
	for (int_t k = 0; k < nsupers; ++k)
	{
        if(supernodeMask[k]==1)
        {
            int_t krow = PROW (k, grid);
            int_t kcol = PCOL (k, grid);

            if (mycol == kcol)
            {
                int_t lk = LBj(k, grid);
                int_t flag=0;
                int_t* lsub = Lrowind_bc_ptr[lk];
                int_t* lloc = Lindval_loc_bc_ptr[lk];
                if(lsub){
                nb = lsub[0];
                if(nb>0){
                    if(myrow!=krow){
                        idx_i = nb;
                        for (lb=0;lb<nb;lb++){
                            lptr1_tmp = lloc[lb+idx_i];
                            ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                            if(supernodeMask[ik]>0)
                                flag=1;
                        }
                    }
                    if(flag==1)
                        nfrecvx++;
                }
                }
            }
        }
	}

	return nfrecvx;
}

int getNfrecvxLeaf(sForest_t* sforest, int_t** Lrowind_bc_ptr, gridinfo_t * grid)
{
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);

	int_t nnodes =   sforest->nNodes ;
	int_t *nodeList = sforest->nodeList ;
	int nfrecvx = 0;
	for (int_t k0 = 0; k0 < nnodes; ++k0)
	{
		int_t k = nodeList[k0];
		int_t krow = PROW (k, grid);
		int_t kcol = PCOL (k, grid);

		if (mycol == kcol && myrow != krow)
		{
			int_t lk = LBj(k, grid);
			int_t* lsub = Lrowind_bc_ptr[lk];
			if (lsub)
			{
				if (lsub[0] > 0)
				{
					/* code */
					nfrecvx++;
				}
			}

		}
	}

	return nfrecvx;
}

int* getfmod_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, int_t** Lrowind_bc_ptr, int_t** Lindval_loc_bc_ptr, gridinfo_t * grid)
{
	int* fmod;
	int_t iam = grid->iam;
	int_t myrow = MYROW (iam, grid);
	int_t mycol = MYCOL (iam, grid);
    int_t nb;

    int_t idx_n,idx_i, lptr1_tmp, lb, lk, ik, ljk;

	if (!(fmod = int32Calloc_dist (nlb)))
		ABORT ("Calloc fails for fmod[].");

	for (int_t k = 0; k < nsupers; ++k)
	{
		if(supernodeMask[k]>0)
        {
            int_t krow = PROW (k, grid);
            int_t kcol = PCOL (k, grid);

            if (mycol == kcol)
            {
                ljk = LBj(k, grid);
                int_t* lsub = Lrowind_bc_ptr[ljk];
                int_t* lloc = Lindval_loc_bc_ptr[ljk];
                if(lsub){
                if(lsub[0]>0){
                    if(myrow==krow){
                        nb = lsub[0] - 1;
                        idx_n = 1;
                        idx_i = nb+2;
                    }else{
                        nb = lsub[0];
                        idx_n = 0;
                        idx_i = nb;
                    }
                    for (lb=0;lb<nb;lb++){
                        lk = lloc[lb+idx_n]; /* Local block number, row-wise. */
                        lptr1_tmp = lloc[lb+idx_i];
                        ik = lsub[lptr1_tmp]; /* Global block number, row-wise. */
                        if(supernodeMask[ik]>0)
                            fmod[lk] +=1;
                    }
                }
                }

            }
        }
	}

	// for (int_t i = 0; i < nlb; ++i){
    //     printf("id %5d i %5d fmod %5d\n",iam,i,fmod[i]);
    // }

	return fmod;
}


int* getfmodLeaf(int_t nlb, int* fmod_i)
{
	int* fmod;
	if (!(fmod = int32Calloc_dist(nlb)))
		ABORT ("Calloc fails for fmod[].");
	for (int_t i = 0; i < nlb; ++i)
		fmod[i] = fmod_i[i];

	return fmod;
}


int getldu(int_t knsupc, int_t iklrow, int_t* usub )
{
    int ldu = 0;

    for (int_t jj = 0; jj < knsupc; ++jj)
    {
        int_t fnz = usub[jj];
        if ( fnz < iklrow )
        {
            int segsize = iklrow - fnz;
            ldu = SUPERLU_MAX(ldu, segsize);
        }

    }
    return ldu;
}


int* getBmod3d(int_t treeId, int_t nlb, sForest_t* sforest, int_t* xsup,int_t **Ufstnz_br_ptr,
                 int_t* supernode2treeMap, gridinfo_t * grid)
{
    // Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    // dLocalLU_t *Llu = LUstruct->Llu;
    // int_t* xsup = Glu_persist->xsup;
    int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    if (nnodes < 1) return NULL;
    int_t *nodeList = sforest->nodeList ;

    // int_t Pr = grid->nprow;
    // int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    // int_t mycol = MYCOL (iam, grid);
    // int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    int* bmod = SUPERLU_MALLOC(sizeof(int) * nlb);

    for (int_t k = 0; k < nlb; ++k)
        bmod[k] = 0;
    for (int_t k0 = 0; k0 < nnodes ; ++k0)
    {
        /* code */

        int_t k = nodeList[k0];

        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid);
            bmod[lk] = 0;
            int_t* usub = Ufstnz_br_ptr[lk];
            if (usub)
            {
                /* code */
                int_t nub = usub[0];       /* Number of blocks in the block row U(k,:) */
                int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
                // int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
                for (int_t ii = 0; ii < nub; ii++)
                {
                    int_t jb = usub[iukp];
                    if ( supernode2treeMap[jb] == treeId)
                    {
                        /* code */
                        bmod[lk]++;
                    }
                    iukp += UB_DESCRIPTOR;
                    iukp += SuperSize (jb);

                }


            }
            else
            {
                bmod[lk] = 0;
            }

        }
    }
    return bmod;
}

int* getBmod3d_newsolve(int_t nlb, int_t nsupers, int* supernodeMask, int_t* xsup, int_t **Ufstnz_br_ptr,
                  gridinfo_t * grid)
{
    // Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
    // dLocalLU_t *Llu = LUstruct->Llu;
    // int_t* xsup = Glu_persist->xsup;
    // int_t nnodes =   sforest->nNodes ;      // number of nodes in the tree
    // if (nnodes < 1) return NULL;
    // int_t *nodeList = sforest->nodeList ;

    // int_t Pr = grid->nprow;
    // int_t Pc = grid->npcol;
    int_t iam = grid->iam;
    int_t myrow = MYROW (iam, grid);
    // int_t mycol = MYCOL (iam, grid);
    // int_t **Ufstnz_br_ptr = Llu->Ufstnz_br_ptr;
    int* bmod = SUPERLU_MALLOC(sizeof(int) * nlb);

    for (int_t k = 0; k < nlb; ++k)
        bmod[k] = 0;

	for (int_t k = 0; k < nsupers; ++k)
	{
		if(supernodeMask[k]>0)
        {
        int_t krow = PROW (k, grid);
        if (myrow == krow)
        {
            int_t lk = LBi (k, grid);
            bmod[lk] = 0;
            int_t* usub = Ufstnz_br_ptr[lk];
            if (usub)
            {
                /* code */
                int_t nub = usub[0];       /* Number of blocks in the block row U(k,:) */
                int_t iukp = BR_HEADER;   /* Skip header; Pointer to index[] of U(k,:) */
                // int_t rukp = 0;           /* Pointer to nzval[] of U(k,:) */
                for (int_t ii = 0; ii < nub; ii++)
                {
                    int_t jb = usub[iukp];
                    if(supernodeMask[jb]>0)
                    {
                        /* code */
                        bmod[lk]++;
                    }
                    iukp += UB_DESCRIPTOR;
                    iukp += SuperSize (jb);

                }
            }
            else
            {
                bmod[lk] = 0;
            }

        }
        }
    }
    return bmod;
}




// #ifdef HAVE_NVSHMEM   
/*global variables for nvshmem, is it safe to be put them here? */
int* mystatus, *mystatusmod,*d_rownum,*d_rowstart;
int* mystatus_u, *mystatusmod_u;
int *d_status, *d_statusmod;
uint64_t *flag_bc_q, *flag_rd_q ;
int* my_flag_bc, *my_flag_rd;
int* d_mynum,*d_mymaskstart,*d_mymasklength;
int* d_mynum_u,*d_mymaskstart_u,*d_mymasklength_u;
int*d_nfrecv, *h_nfrecv, *d_colnum;
int*d_nfrecv_u, *h_nfrecv_u, *d_colnum_u;
int* d_nfrecvmod, *h_nfrecvmod, *d_colnummod;
int* d_nfrecvmod_u, *h_nfrecvmod_u, *d_colnummod_u;
int* d_mynummod,*d_mymaskstartmod,*d_mymasklengthmod;
int* d_mynummod_u,*d_mymaskstartmod_u,*d_mymasklengthmod_u;
int *d_recv_cnt, *d_msgnum;
int *d_recv_cnt_u, *d_msgnum_u;
int *d_flag_mod, *d_flag_mod_u;
// #endif