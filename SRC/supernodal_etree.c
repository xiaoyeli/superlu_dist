/*! @file
 * \brief function to generate supernodal etree
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
//#include "supernodal_etree.h"

#define INT_T_ALLOC(x)  ((int_t *) SUPERLU_MALLOC ( (x) * sizeof (int_t)))
int_t log2i(int_t index)
{
	int_t targetlevel = 0;
	while (index >>= 1) ++targetlevel;
	return targetlevel;
}

/**
 * Returns Supernodal Elimination Tree
 * @param  nsuper Number of Supernodes
 * @param  etree  Scalar elimination tree 
 * @param  supno  Vertex to supernode mapping
 * @param  xsup   Supernodal boundaries
 * @return       Supernodal elimination tree
 */
int_t *supernodal_etree(int_t nsuper, int_t * etree, int_t* supno, int_t *xsup)
{
    //	int_t *setree = malloc(sizeof(int_t) * nsuper);
    int_t *setree = intMalloc_dist(nsuper);  // Sherry fix
	/*initialzing the loop*/
	for (int i = 0; i < nsuper; ++i)
	{
	    setree[i] = nsuper;
	}
	/*calculating the setree*/
	for (int i = 0; i < nsuper - 1; ++i)
	{
	    int_t ftree = etree[xsup[i + 1] - 1];
	    if (ftree < xsup[nsuper])
		{
		    setree[i] = supno[etree[xsup[i + 1] - 1]];
		}
	}
	return setree;
}
/*takes supernodal elimination tree and for each
supernode calculates "level" in elimination tree*/
int_t* topological_ordering(int_t nsuper, int_t* setree)
{
    // int_t *tsort_setree = malloc(sizeof(int_t) * nsuper);
    int_t *tsort_setree = intMalloc_dist(nsuper); // Sherry fix
	for (int i = 0; i < nsuper; ++i)
	{
	    tsort_setree[i] = 0; /*initializing all levels to zero*/
	}
	for (int i = 0; i < nsuper - 1; ++i)
	{
	    /*level of parent = MAX(level_of_children()+1)*/
	    tsort_setree[setree[i]] = SUPERLU_MAX(tsort_setree[setree[i]], tsort_setree[i] + 1);
	}
	return tsort_setree;
}


treeList_t* setree2list(int_t nsuper, int_t* setree )
{
    treeList_t* treeList = (treeList_t* ) SUPERLU_MALLOC (sizeof(treeList_t) * (nsuper + 1));

	// initialize the struct
	for (int i = 0; i < nsuper + 1; ++i)
	{
	    treeList[i].numChild = 0;
	    treeList[i].numDescendents = 1;	/*numdescen includes myself*/
	    treeList[i].left = -1;
	    treeList[i].right = -1;
	    treeList[i].right = -1;
	    treeList[i].depth = 0;
	}
	for (int i = 0; i < nsuper; ++i)
	{
	    // updating i-th supernodes parents
	    int_t parenti = setree[i];
	    treeList[parenti].numDescendents +=  treeList[i].numDescendents;
	    treeList[parenti].numChild++;
	}

	/*allocate memory for children lists*/
	for (int i = 0; i < nsuper + 1; ++i)
	{
	    //treeList[i].childrenList = INT_T_ALLOC (treeList[i].numChild);
	    treeList[i].childrenList = intMalloc_dist(treeList[i].numChild);
	    treeList[i].numChild = 0;
	}

	for (int i = 0; i < nsuper; ++i)
	{
	    // updating i-th supernodes parents
	    int_t parenti = setree[i];
	    treeList[parenti].childrenList[treeList[parenti].numChild] = i;
	    treeList[parenti].numChild++;
	}

	return treeList;

} /* setree2list */

// Sherry added 
int  free_treelist(int_t nsuper, treeList_t* treeList)
{
    for (int i = 0; i < nsuper + 1; ++i) {
	SUPERLU_FREE(treeList[i].childrenList);
    }
    SUPERLU_FREE(treeList);
    return 0;
}

int_t estimateWeight(int_t nsupers, int_t*setree, treeList_t* treeList, int_t* xsup)
{
	if (getenv("WF"))
	{
		if (strcmp(getenv("WF"), "One" ) == 0)
		{
			for (int i = 0; i < nsupers; ++i)
			{
				treeList[i].weight = 1.0;
			}
		}
		else if (strcmp(getenv("WF"), "Ns" ) == 0)
		{
			for (int i = 0; i < nsupers; ++i)
			{
				double sz = 1.0 *  SuperSize(i);
				treeList[i].weight = sz;
			}
		}
		else if (strcmp(getenv("WF"), "NsDep" ) == 0)
		{
			for (int i = 0; i < nsupers; ++i)
			{
				double dep = 1.0 * treeList[i].depth ;
				double sz = 1.0 *  SuperSize(i);
				treeList[i].weight = sz * dep;
			}
		}
		else if (strcmp(getenv("WF"), "NsDep2" ) == 0)
		{
			for (int i = 0; i < nsupers; ++i)
			{
				double dep = 1.0 * treeList[i].depth ;
				double sz = 1.0 *  SuperSize(i);
				treeList[i].weight = 3 * sz * dep * (sz + dep) + sz * sz * sz ;

			}

		}
		else
		{
			for (int i = 0; i < nsupers; ++i)
			{
				treeList[i].weight = treeList[i].scuWeight;
			}
		}
	}
	else
	{
		for (int i = 0; i < nsupers; ++i)
		{
			treeList[i].weight = treeList[i].scuWeight;

		}
	}

	return 0;
} /* estimateWeight */


int_t calcTreeWeight(int_t nsupers, int_t*setree, treeList_t* treeList, int_t* xsup)
{

	// initializing naive weight
	for (int i = 0; i < nsupers; ++i)
	{
		treeList[i].depth = 0;
	}

	for (int i = nsupers - 1; i > -1; --i)
	{
		/* code */
		int_t myDep = treeList[i].depth;
		for (int cIdx = 0; cIdx < treeList[i].numChild; ++cIdx)
		{
			/* code */
			int_t child = treeList[i].childrenList[cIdx];
			treeList[child].depth = myDep + SuperSize(i) ;

		}
	}


	// for (int i = 0; i < nsupers; ++i)
	// {

	// 	// treeList[i].weight = 1.0 * treeList[i].numDescendents;
	// 	double dep = 1.0 * treeList[i].depth ;
	// 	double sz = 1.0 *  SuperSize(i);
	// 	treeList[i].weight = 1.0;
	// 	treeList[i].weight = sz;
	// 	treeList[i].weight = sz * sz * sz;
	// 	treeList[i].weight = 3 * sz * dep * (sz + dep) + sz * sz * sz ;
	// 	treeList[i].weight = treeList[i].scuWeight;
	// 	// treeList[i].treeWeight = treeList[i].weight;
	// 	// treeList[i].depth = 0;
	// }

	estimateWeight(nsupers, setree, treeList, xsup);

	for (int i = 0; i < nsupers; ++i)
	{
		treeList[i].iWeight = treeList[i].weight;
	}


	for (int i = 0; i < nsupers; ++i)
	{
		int_t parenti = setree[i];
		treeList[parenti].iWeight += treeList[i].iWeight;
	}


	return 0;

} /* calcTreeWeight */


int_t printFileList(char* sname, int_t nnodes, int_t*dlist, int_t*setree)
{
	FILE* fp = fopen(sname, "w");
	/*beginning of the file */
	fprintf(fp, "//dot file generated by pdgstrf\n");
	fprintf(fp, "digraph elimination_tree {\n");
	for (int i = 0; i < nnodes; ++i)
	{
	    /* code */
	  fprintf(fp, IFMT " -> " IFMT ";\n", dlist[i], setree[dlist[i]]);
	}
	/*end of the file */
	fprintf(fp, "}\n");
	fprintf(fp, "//EOF\n");
	fclose(fp);
	return 0;
}

int_t getDescendList(int_t k, int_t*dlist,  treeList_t* treeList)
// post order traversal
{
	if (k < 0) return 0;

	int_t cDesc = 0;

	for (int_t child = 0; child < treeList[k].numChild; ++child)
	{
		/* code */
		int_t nChild = treeList[k].childrenList[child];
		cDesc += getDescendList(nChild, dlist + cDesc, treeList);
	}

	dlist[cDesc] = k;
	return cDesc + 1;
}


int_t getCommonAncsCount(int_t k, treeList_t* treeList)
{
	// given a supernode k, give me the list of ancestors nodes
	int_t cur = k;
	int_t count = 1;
	while (treeList[cur].numChild == 1)
	{
		cur = treeList[cur].childrenList[0];
		count++;
	}
	return count;
}
int_t getCommonAncestorList(int_t k, int_t* alist,  int_t* seTree, treeList_t* treeList)
{
	// given a supernode k, give me the list of ancestors nodes
	int_t cur = k;
	int_t count = 1;
	while (treeList[cur].numChild == 1)
	{
		cur = treeList[cur].childrenList[0];
		count++;
	}


	alist[0] = cur;
	for (int i = 1; i < count; ++i)
	{
		/* code */
		alist[i] = seTree[cur];
		cur = seTree[cur];
	}
	return count;
}

int cmpfunc (const void * a, const void * b)
{
	return ( *(int_t*)a - * (int_t*)b );
}

int_t* getPermNodeList(int_t nnode, 	// number of nodes
                       int_t* nlist, int_t* perm_c_sup, int_t* iperm_c_sup)
//from list of nodes, get permutation of factorization
{
	int_t* perm_l = (int_t* ) SUPERLU_MALLOC(sizeof(int_t) * nnode);
	int_t* iperm_l = (int_t* ) SUPERLU_MALLOC(sizeof(int_t) * nnode);
	for (int_t i = 0; i < nnode; ++i)
	{
		/* code */
		// printf("%d %d %d\n",i, nlist[i],iperm_c_sup[nlist[i]] );
		iperm_l[i] = iperm_c_sup[nlist[i]]; //order of factorization
	}
	qsort(iperm_l, nnode, sizeof(int_t), cmpfunc);

	for (int_t i = 0; i < nnode; ++i)
	{
		/* code */
		perm_l[i] = perm_c_sup[iperm_l[i]]; //order of factorization
	}
	SUPERLU_FREE(iperm_l);
	return perm_l;
}
int_t* getEtreeLB(int_t nnodes, int_t* perm_l, int_t* gTopOrder)
// calculates EtreeLB boundaries for given list of nodes, via perm_l
{
	//calculate minimum and maximum topOrder
	int minTop, maxTop;
	minTop = gTopOrder[perm_l[0]];
	maxTop = gTopOrder[perm_l[nnodes - 1]];
	int numLB = maxTop - minTop + 2;
	//int_t* lEtreeLB = (int_t *) malloc( sizeof(int_t) * numLB);
	int_t* lEtreeLB = (int_t *) intMalloc_dist(numLB); // Sherry fix
	for (int i = 0; i < numLB; ++i)
	{
		/* initalize */
		lEtreeLB[i] = 0;
	}
	lEtreeLB[0] = 0;
	int curLevel = minTop;
	int curPtr = 1;
	for (int i = 0; i < nnodes ; ++i)
	{
		/* code */
		if (curLevel != gTopOrder[perm_l[i]])
		{
			/* creset */
			curLevel = gTopOrder[perm_l[i]];
			lEtreeLB[curPtr] = i;
			curPtr++;
		}
	}
	lEtreeLB[curPtr] = lEtreeLB[curPtr - 1] + 1;
	printf("numLB=%d curPtr=%d \n", numLB, curPtr);
	for (int i = 0; i < numLB; ++i)
	{
	    printf(IFMT, lEtreeLB[i]);
	}

	return lEtreeLB;
}

int_t* getSubTreeRoots(int_t k, treeList_t* treeList)
{
	int_t* srootList = (int_t* ) SUPERLU_MALLOC(sizeof(int_t) * 2);
	int_t cur = k;
	while (treeList[cur].numChild == 1 && cur > 0)
	{
		cur = treeList[cur].childrenList[0];
	}

	if (treeList[cur].numChild == 2)
	{
		/* code */
		srootList[0] = treeList[cur].childrenList[0];
		srootList[1] = treeList[cur].childrenList[1];
		// printf("Last node =%d, numchilds=%d,  desc[%d] = %d, desc[%d] = %d \n ",
		// 	cur, treeList[cur].numChild,
		// 	srootList[0], treeList[srootList[0]].numDescendents,
		// 	srootList[1], treeList[srootList[1]].numDescendents );
	}
	else
	{
		/* code */
		srootList[0] = -1;
		srootList[1] = -1;
	}

	return srootList;
}

int_t testSubtreeNodelist(int_t nsupers, int_t numList, int_t** nodeList, int_t* nodeCount)
// tests disjoint and union
{
    //int_t* slist = (int_t* ) malloc(sizeof(int_t) * nsupers);
    int_t* slist = intMalloc_dist(nsupers); // Sherry fix
	/*intialize each entry with zero */
	for (int_t i = 0; i < nsupers; ++i)
	{
		/* code */
		slist[i] = 0;
	}
	for (int_t list = 0; list < numList; ++list)
	{
		/* code */
		for (int_t nd = 0; nd < nodeCount[list]; ++nd)
		{
			slist[nodeList[list][nd]]++;
		}
	}

	for (int_t i = 0; i < nsupers; ++i)
	{
		/* code */
		assert(slist[i] == 1);
	}
	printf("testSubtreeNodelist Passed\n");
	SUPERLU_FREE(slist);
	return 0;
}
int_t testListPerm(int_t nodeCount, int_t* nodeList, int_t* permList, int_t* gTopLevel)
{
	// checking monotonicity
	for (int i = 0; i < nodeCount - 1; ++i)
	{
	    if (!( gTopLevel[permList[i]] <= gTopLevel[permList[i + 1]]))
	      {
		  /* code */
		printf("%d :" IFMT "(" IFMT ")" IFMT "(" IFMT ")\n", i,
		       permList[i], gTopLevel[permList[i]],
		       permList[i + 1], gTopLevel[permList[i + 1]] );
	      }
	    assert( gTopLevel[permList[i]] <= gTopLevel[permList[i + 1]]);
	}
#if 0
	int_t* slist = (int_t* ) malloc(sizeof(int_t) * nodeCount);
	int_t* plist = (int_t* ) malloc(sizeof(int_t) * nodeCount);
#else
	int_t* slist = intMalloc_dist(nodeCount);
	int_t* plist = intMalloc_dist(nodeCount);
#endif
	// copy lists
	for (int_t i = 0; i < nodeCount; ++i)
	{
		slist[i] = nodeList[i];
		plist[i] = permList[i];
	}
	// sort them
	qsort(slist, nodeCount, sizeof(int_t), cmpfunc);
	qsort(plist, nodeCount, sizeof(int_t), cmpfunc);
	for (int_t i = 0; i < nodeCount; ++i)
	{
		assert( slist[i] == plist[i]);
	}
	printf("permList Test Passed\n");

	SUPERLU_FREE(slist);
	SUPERLU_FREE(plist);

	return 0;
}


int_t mergPermTest(int_t nperms, int_t* gperms, int_t* nnodes);

// Sherry: the following routine is not called ??
int_t* merg_perms(int_t nperms, int_t* nnodes, int_t** perms)
{
	// merges three permutations
	int_t nn = 0;
	//add permutations
	for (int i = 0; i < nperms; ++i)
	{
		nn += nnodes[i];
	}

	// alloc address
	//int_t* gperm = (int_t*) malloc(nn * sizeof(int_t));
	int_t* gperm = intMalloc_dist(nn);  // Sherry fix

	//now concatenat arrays
	int ptr = 0;
	for (int tr = 0; tr < nperms; ++tr)
	{
	    /* code */
	    for (int nd = 0; nd < nnodes[tr]; ++nd)
	      {
		/* code */
		gperm[ptr] = perms[tr][nd];
		printf("%d %d %d" IFMT "\n", tr, ptr, nd, perms[tr][nd] );
		ptr++;
	      }
	}
	mergPermTest( nperms, gperm, nnodes);
	return gperm;
} /* merg_perms */

int_t mergPermTest(int_t nperms, int_t* gperms, int_t* nnodes)
{
	// merges three permutations
	int_t nn = 0;
	//add permutations
	for (int i = 0; i < nperms; ++i)
	{
		nn += nnodes[i];
	}

	// alloc address
	// int_t* tperm = (int_t*) malloc(nn * sizeof(int_t));
	int_t* tperm = intMalloc_dist(nn);  // Sherry fix

	for (int i = 0; i < nn; ++i)
	{
		tperm[i] = 0;
	}
	for (int i = 0; i < nn; ++i)
	{
		/* code */
		printf("%d" IFMT "\n", i, gperms[i] );
		tperm[gperms[i]]++;
	}
	for (int i = 0; i < nn; ++i)
	{
		/* code */
		assert(tperm[i] == 1);
	}
	SUPERLU_FREE(tperm);
	return nn;
} /* mergPermTest */

#if 0 // Sherry: not called anymore
int* getLastDep(gridinfo_t *grid, SuperLUStat_t *stat,
		superlu_dist_options_t *options,
                LocalLU_t *Llu, int_t* xsup,
                int_t num_look_aheads, int_t nsupers, int_t * iperm_c_supno)
{
	/* constructing look-ahead table to indicate the last dependency */
	int_t iam = grid->iam;
	int_t Pc = grid->npcol;
	int_t  Pr = grid->nprow;
	int_t  myrow = MYROW (iam, grid);
	int_t  mycol = MYCOL (iam, grid);
	int_t   ncb = nsupers / Pc;
	int_t  nrb = nsupers / Pr;
	stat->num_look_aheads = num_look_aheads;
	int* look_ahead_l = SUPERLU_MALLOC (nsupers * sizeof (int));
	int* look_ahead = SUPERLU_MALLOC (nsupers * sizeof (int));
	for (int_t lb = 0; lb < nsupers; lb++)
		look_ahead_l[lb] = -1;
	/* go through U-factor */
	for (int_t lb = 0; lb < nrb; ++lb)
	{
		int_t ib = lb * Pr + myrow;
		int_t* index = Llu->Ufstnz_br_ptr[lb];
		if (index)              /* Not an empty row */
		{
			int_t k = BR_HEADER;
			for (int_t j = 0; j < index[0]; ++j)
			{
				int_t jb = index[k];
				if (jb != ib)
					look_ahead_l[jb] =
					    SUPERLU_MAX (iperm_c_supno[ib], look_ahead_l[jb]);
				k += UB_DESCRIPTOR + SuperSize (index[k]);
			}
		}
	}
	if (myrow < nsupers % grid->nprow)
	{
		int_t ib = nrb * Pr + myrow;
		int_t*  index = Llu->Ufstnz_br_ptr[nrb];
		if (index)              /* Not an empty row */
		{
			int_t k = BR_HEADER;
			for (int_t j = 0; j < index[0]; ++j)
			{
				int_t jb = index[k];
				if (jb != ib)
					look_ahead_l[jb] =
					    SUPERLU_MAX (iperm_c_supno[ib], look_ahead_l[jb]);
				k += UB_DESCRIPTOR + SuperSize (index[k]);
			}
		}
	}
	if (options->SymPattern == NO)
	{
		/* go through L-factor */
		for (int_t lb = 0; lb < ncb; lb++)
		{
			int_t ib = lb * Pc + mycol;
			int_t* index = Llu->Lrowind_bc_ptr[lb];
			if (index)
			{
				int_t k = BC_HEADER;
				for (int_t j = 0; j < index[0]; j++)
				{
					int_t jb = index[k];
					if (jb != ib)
						look_ahead_l[jb] =
						    SUPERLU_MAX (iperm_c_supno[ib], look_ahead_l[jb]);
					k += LB_DESCRIPTOR + index[k + 1];
				}
			}
		}
		if (mycol < nsupers % grid->npcol)
		{
			int_t ib = ncb * Pc + mycol;
			int_t* index = Llu->Lrowind_bc_ptr[ncb];
			if (index)
			{
				int_t k = BC_HEADER;
				for (int_t j = 0; j < index[0]; j++)
				{
					int_t jb = index[k];
					if (jb != ib)
						look_ahead_l[jb] =
						    SUPERLU_MAX (iperm_c_supno[ib], look_ahead_l[jb]);
					k += LB_DESCRIPTOR + index[k + 1];
				}
			}
		}
	}
	MPI_Allreduce (look_ahead_l, look_ahead, nsupers, MPI_INT, MPI_MAX,
	               grid->comm);
	SUPERLU_FREE (look_ahead_l);
	return look_ahead;
}

int* getLastDepBtree( int_t nsupers, treeList_t* treeList)
{
	int* look_ahead = SUPERLU_MALLOC (nsupers * sizeof (int));
	for (int i = 0; i < nsupers; ++i)
	{
		look_ahead[i] = -1;
	}
	for (int k = 0; k < nsupers; ++k)
	{
		/* code */
		for (int_t child = 0; child < treeList[k].numChild; ++child)
		{
			/* code */
			switch ( child)
			{
			case 0:
				look_ahead[k] = SUPERLU_MAX(look_ahead[k], treeList[k].left);
				break;
			case 1:
				look_ahead[k] = SUPERLU_MAX(look_ahead[k], treeList[k].right);
				break;
			case 2:
				look_ahead[k] = SUPERLU_MAX(look_ahead[k], treeList[k].extra);
				break;
			default:
				break;
			}
		}
	}
	return look_ahead;
}

#endif // Sherry: not called anymore


int_t* getGlobal_iperm(int_t nsupers, int_t nperms,  // number of permutations
                       int_t** perms, 		// array of permutations
                       int_t* nnodes 		// number of nodes in each permutation
                      )
{
	int_t*  gperm = SUPERLU_MALLOC (nsupers * sizeof (int_t));
	int_t*  giperm = SUPERLU_MALLOC (nsupers * sizeof (int_t));
	int_t ptr = 0;
	for (int_t perm = 0; perm < nperms; ++perm)
	{
		/* code */
		for (int_t node = 0; node < nnodes[perm]; ++node)
		{
			/* code */
			gperm[ptr] = perms[perm][node];
			ptr++;
		}
	}
	assert(ptr == nsupers);
	for (int_t i = 0; i < nsupers; ++i)
	{
		giperm[gperm[i]] = i;
	}
	SUPERLU_FREE(gperm);
	return giperm;
}
int_t* getTreeHeads(int_t maxLvl, int_t nsupers, treeList_t* treeList)
{
	int_t numTrees = (1 << maxLvl) - 1;
	int_t* treeHeads = SUPERLU_MALLOC (numTrees * sizeof (int_t));
	// for (int i = 0; i < numTrees; ++i)
	// {
	// 	/* code */
	// 	treeHeads[i]=0;
	// }
	treeHeads[0] = nsupers - 1;
	for (int_t lvl = 0; lvl < maxLvl - 1; ++lvl)
	{
		/* code */
		int_t st = (1 << lvl) - 1;
		int_t end = 2 * st + 1;
		for (int_t i = st; i < end; ++i)
		{
			/* code */
			int_t * sroots;
			sroots = getSubTreeRoots(treeHeads[i], treeList);
			treeHeads[2 * i + 1] = sroots[0];
			treeHeads[2 * i + 2] = sroots[1];
			SUPERLU_FREE(sroots);
		}
	}
	return treeHeads;
}

int_t* calcNumNodes(int_t maxLvl,  int_t* treeHeads, treeList_t* treeList)
{
	int_t numTrees = (1 << maxLvl) - 1;
	int_t* nnodes = SUPERLU_MALLOC (numTrees * sizeof (int_t));
	for (int_t i = 0; i < numTrees; ++i)
	{
		/* code */
		if (treeHeads[i] > -1)
		{
			/* code */
			nnodes[i] = treeList[treeHeads[i]].numDescendents;
		}
		else
		{
			nnodes[i] = 0;
		}

	}
	for (int_t i = 0; i < numTrees / 2 ; ++i)
	{
		/* code */
		nnodes[i] -= (nnodes[2 * i + 1] + nnodes[2 * i + 2]);
	}
	return nnodes;
}

int_t** getNodeList(int_t maxLvl, int_t* setree, int_t* nnodes,
                    int_t* treeHeads, treeList_t* treeList)
{
	int_t numTrees = (1 << maxLvl) - 1;
	int_t** nodeList = SUPERLU_MALLOC (numTrees * sizeof (int_t*));
	for (int_t i = 0; i < numTrees; ++i)
	{
		/* code */
		if (nnodes[i] > 0)
		{
			nodeList[i] = SUPERLU_MALLOC (nnodes[i] * sizeof (int_t));
			assert(nodeList[i]);
		}
		else
		{
			nodeList[i] = NULL;
		}

	}

	for (int_t lvl = 0; lvl < maxLvl - 1; ++lvl)
	{
		/* code */
		int_t st = (1 << lvl) - 1;
		int_t end = 2 * st + 1;
		for (int_t i = st; i < end; ++i)
		{
			/* code */
			if (nodeList[i])
				getCommonAncestorList(treeHeads[i], nodeList[i],  setree, treeList);
		}
	}

	int_t st = (1 << (maxLvl - 1)) - 1;
	int_t end = 2 * st + 1;
	for (int_t i = st; i < end; ++i)
	{
		/* code */
		getDescendList(treeHeads[i], nodeList[i],  treeList);
	}
	return nodeList;
}

int_t* getGridTrees( gridinfo3d_t* grid3d)
{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t* myTreeIdx = (int_t*) SUPERLU_MALLOC (maxLvl * sizeof (int_t));
	myTreeIdx[0] = grid3d->zscp.Np - 1 + grid3d->zscp.Iam ;
	for (int i = 1; i < maxLvl; ++i)
	{
		/* code */
		myTreeIdx[i] = (myTreeIdx[i - 1] - 1) / 2;
	}
	return myTreeIdx;
}

int_t* getReplicatedTrees( gridinfo3d_t* grid3d)
{
	int_t maxLvl = log2i(grid3d->zscp.Np) + 1;
	int_t* myZeroTrIdxs = (int_t*) SUPERLU_MALLOC (maxLvl * sizeof (int_t));
	for (int i = 0; i < maxLvl; ++i)
	{
		/* code */
		if (grid3d->zscp.Iam %  (1 << i) )
		{
			myZeroTrIdxs[i] = 1;
		}
		else
		{
			myZeroTrIdxs[i] = 0;
		}
	}
	return myZeroTrIdxs;
}


int_t* getMyIperm(int_t nnodes, int_t nsupers, int_t* myPerm)
{
	if (nnodes < 0) return NULL;
	int_t* myIperm =  INT_T_ALLOC(nsupers);
	for (int_t i = 0; i < nsupers; ++i)
	{
		/* code */
		myIperm[i] = -1;
	}
	for (int_t i = 0; i < nnodes; ++i)
	{
		/* code */
		assert(myPerm[i] < nsupers);
		myIperm[myPerm[i]] = i;
	}
	return myIperm;
}
int_t* getMyTopOrder(int_t nnodes, int_t* myPerm, int_t* myIperm, int_t* setree )
{
	if (nnodes < 0) return NULL;
	int_t* myTopOrder =  INT_T_ALLOC(nnodes);
	for (int_t i = 0; i < nnodes; ++i)
	{
		myTopOrder[i] = 0; /*initializing all levels to zero*/
	}
	for (int_t i = 0; i < nnodes - 1; ++i)
	{
		/*level of parent = MAX(level_of_children()+1)*/
		int_t inode = myPerm[i];
		int_t iparent = setree[inode];
		int_t iparentIdx  = myIperm[iparent];
		// if(iparentIdx >= nnodes) printf("%d %d %d %d \n", inode, iparent, nnodes, iparentIdx);
		// assert(iparentIdx < nnodes);
		// if (iparentIdx != -1)
		if (0<= iparentIdx && iparentIdx<nnodes )
			/*if my parent is in my tree only*/
		{
			myTopOrder[iparentIdx] = SUPERLU_MAX(myTopOrder[iparentIdx], myTopOrder[i] + 1);
		}
	}
	return myTopOrder;
}
int_t checkConsistancyPermTopOrder(int_t nnodes, int_t* myTopOrder)
/*Ideally top order should be monotonically increasing*/
{
	for (int_t i = 0; i < nnodes - 1; ++i)
	{
		assert(myTopOrder[i] <= myTopOrder[i + 1]);
	}
	return 0;
}
int_t* getMyEtLims(int_t nnodes, int_t* myTopOrder)
{
	if (nnodes < 0) return NULL;
	checkConsistancyPermTopOrder(nnodes, myTopOrder);
	int_t numLvl = myTopOrder[nnodes - 1] + 1;
	int_t* myTopLims = INT_T_ALLOC(numLvl + 1);
	for (int i = 0; i < numLvl + 1; ++i)
	{
		myTopLims[i] = 0;
	}
	int_t nxtLvl = 1;
	for (int_t i = 0; i < nnodes - 1; ++i)
	{
		/* code */
		if (myTopOrder[i] != myTopOrder[i + 1])
		{
			/* code */
			myTopLims[nxtLvl] = i + 1;
			nxtLvl++;
		}
	}
	assert(nxtLvl == numLvl);
	myTopLims[numLvl] = nnodes;

	return myTopLims;
}

treeTopoInfo_t getMyTreeTopoInfo(int_t nnodes, int_t  nsupers,
                                 int_t* myPerm, int_t* setree)
{
	treeTopoInfo_t ttI;
	int_t* myIperm = getMyIperm(nnodes, nsupers, myPerm);
	int_t* myTopOrder =  getMyTopOrder(nnodes, myPerm, myIperm, setree );
	ttI.myIperm = myIperm;
	ttI.numLvl = myTopOrder[nnodes - 1] + 1;
	ttI.eTreeTopLims = getMyEtLims(nnodes, myTopOrder);
	return ttI;
}

// Sherry: the following function is not called ??
/*calculated boundries of the topological levels*/
int_t* Etree_LevelBoundry(int_t* perm, int_t* tsort_etree, int_t nsuper)
{
	int_t max_level =  tsort_etree[nsuper - 1] + 1;
	//int_t *Etree_LvlBdry = malloc(sizeof(int_t) * (max_level + 1));
	int_t *Etree_LvlBdry = intMalloc_dist(max_level + 1); // Sherry fix
	Etree_LvlBdry[0] = 0;
	/*calculate end of boundries for each level*/
	for (int_t i = 0; i < max_level; ++i)
	{
		/* code */
		int_t st = 0;
		if (i > 0)
		{
			/* code */
			st = Etree_LvlBdry[i];
		}
		for (int_t j = st; j < nsuper; ++j)
		{
			/* code */
			if (tsort_etree[perm[j]] == i + 1)
			{
				/* code */
				Etree_LvlBdry[i + 1] = j;
				break;
			}
		}
	}
	Etree_LvlBdry[max_level] = nsuper;
	return Etree_LvlBdry;
}

int_t* calculate_num_children(int_t nsuper, int_t* setree)
{
    //int_t* etree_num_children = malloc(sizeof(int_t) * (nsuper));
    int_t* etree_num_children = intMalloc_dist(nsuper); // Sherry fix
	for (int_t i = 0; i < nsuper; ++i)
	{
		/*initialize num children to zero*/
		etree_num_children[i] = 0;
	}
	for (int_t i = 0; i < nsuper; i++)
	{
		if (setree[i] < nsuper)
			etree_num_children[setree[i]]++;
	}
	return etree_num_children;
}
void Print_EtreeLevelBoundry(int_t *Etree_LvlBdry, int_t max_level, int_t nsuper)
{
	for (int i = 0; i < max_level; ++i)
	{
		int st = 0;
		int ed = nsuper;
		st = Etree_LvlBdry[i];
		ed = Etree_LvlBdry[i + 1];
		printf("Level %d, NumSuperNodes=%d,\t Start=%d end=%d\n", i, ed - st, st, ed);
	}
}

void print_etree_leveled(int_t *setree,  int_t* tsort_etree, int_t nsuper)
{
	FILE* fp = fopen("output_sorted.dot", "w");
	int max_level =  tsort_etree[nsuper - 1];
	/*beginning of the file */
	fprintf(fp, "//dot file generated by pdgstrf\n");
	fprintf(fp, "digraph elimination_tree {\n");
	fprintf(fp, "labelloc=\"t\";\n");
	fprintf(fp, "label=\"Depth of the tree is %d\";\n", max_level);

	for (int i = 0; i < nsuper - 1; ++i)
	{
		/* code */
		// fprintf(fp, "%lld -> %lld;\n",iperm[i],iperm[setree[i]]);
		fprintf(fp, "%d -> " IFMT ";\n", i, setree[i]);
	}
	/*adding rank information*/
	for (int i = 0; i < max_level; ++i)
	{
		fprintf(fp, "{ rank=same; ");
		for (int j = 0; j < nsuper; ++j)
		{
			if (tsort_etree[j] == i)
				fprintf(fp, "%d ", j);
		}
		fprintf(fp, "}\n");
	}
	/*end of the file */
	fprintf(fp, "}\n");
	fprintf(fp, "//EOF\n");
	fclose(fp);
}


void printEtree(int_t nsuper, int_t *setree, treeList_t* treeList)
{
	FILE* fp = fopen("output_sorted.dot", "w");
	// int_t max_level =  tsort_etree[nsuper - 1];
	/*beginning of the file */
	fprintf(fp, "//dot file generated by pdgstrf\n");
	fprintf(fp, "digraph elimination_tree {\n");
	// fprintf(fp, "labelloc=\"t\";\n");
	// fprintf(fp, "label=\"Depth of the tree is %d\";\n", max_level);

	for (int i = 0; i < nsuper - 1; ++i)
	{
		/* code */
		// fprintf(fp, "%lld -> %lld;\n",iperm[i],iperm[setree[i]]);
	    fprintf(fp, " \"%d|%d\" -> \"%ld|%ld\";\n", i, (int) treeList[i].depth,
		        (long int) setree[i], (long int) treeList[setree[i]].depth);
	}

	/*end of the file */
	fprintf(fp, "}\n");
	fprintf(fp, "//EOF\n");
	fclose(fp);
}


void print_etree(int_t *setree, int_t* iperm, int_t nsuper)
{
	FILE* fp = fopen("output.dot", "w");
	/*beginning of the file */
	fprintf(fp, "//dot file generated by pdgstrf\n");
	fprintf(fp, "digraph elimination_tree {\n");
	for (int i = 0; i < nsuper; ++i)
	{
	    /* code */
	    fprintf(fp, IFMT " -> " IFMT ";\n", iperm[i], iperm[setree[i]]);
	}
	/*end of the file */
	fprintf(fp, "}\n");
	fprintf(fp, "//EOF\n");
	fclose(fp);
}
