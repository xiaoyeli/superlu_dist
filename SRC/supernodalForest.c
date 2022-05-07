/*! @file
 * \brief SuperLU utilities
 *
 * <pre>
 * -- Distributed SuperLU routine (version 7.0) --
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
// each treehead is an entire subtree (all level below)
#define MAX_TREE_ALLOWED 1024

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

forestPartition_t iterativeFrPartitioning(rForest_t* rforest, int_t nsupers, int_t * setree, treeList_t* treeList)
{

    int_t nTreeSet = rforest->ntrees;
    int_t* treeHeads =  rforest->treeHeads;

    int_t nAnc = 0;
#if 0
    int_t* ancTreeCount = INT_T_ALLOC(MAX_TREE_ALLOWED);
    int_t** ancNodeLists = SUPERLU_MALLOC(MAX_TREE_ALLOWED * sizeof(int_t*));

    double * weightArr = DOUBLE_ALLOC (MAX_TREE_ALLOWED);
    // int_t* treeSet = INT_T_ALLOC(nTreeSet);
    int_t* treeSet = INT_T_ALLOC(MAX_TREE_ALLOWED);
#else  // Sherry fix
    int_t* ancTreeCount = intMalloc_dist(MAX_TREE_ALLOWED);
    int_t** ancNodeLists = SUPERLU_MALLOC(MAX_TREE_ALLOWED * sizeof(int_t*));

    double * weightArr = doubleMalloc_dist(MAX_TREE_ALLOWED);
    int_t* treeSet = intMalloc_dist(MAX_TREE_ALLOWED);
#endif

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
		int_t*  sroots = getSubTreeRoots(MaxTree, treeList);
		if (sroots[0] == -1)
		{
			/* code */
			SUPERLU_FREE(sroots);
			break;
		}

		ancTreeCount[nAnc] = getCommonAncsCount(MaxTree, treeList);
		//int_t * alist = INT_T_ALLOC (ancTreeCount[nAnc]);
		int_t * alist = intMalloc_dist(ancTreeCount[nAnc]);
		getCommonAncestorList(MaxTree, alist, setree, treeList);
		ancNodeLists[nAnc] = alist;
		nAnc++;


		treeSet[idx] = treeSet[nTreeSet - 1];
		weightArr[idx] = treeList[treeSet[idx]].iWeight;
		treeSet[nTreeSet - 1] = sroots[0];
		weightArr[nTreeSet - 1] = treeList[treeSet[nTreeSet - 1]].iWeight;
		treeSet[nTreeSet] = sroots[1];
		weightArr[nTreeSet] = treeList[treeSet[nTreeSet]].iWeight;
		nTreeSet += 1;

		SUPERLU_FREE(sroots);

		if (nTreeSet == MAX_TREE_ALLOWED)
		{
			break;
		}
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
