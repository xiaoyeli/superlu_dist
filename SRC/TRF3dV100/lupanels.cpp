#include "lupanels.hpp"

lpanel_t::lpanel_t(int_t *lsub, double *lval)
{
    return;
}

upanel_t::upanel_t(int_t *lsub, double *uval)
{
    return;
}

int_t upanel_t::packed2skyline(int_t *usub, double *uval)
{
    return 0;
}

LUstruct_v100::LUstruct_v100(int_t nsupers,
                             int_t *isNodeInMyGrid,
                             LUstruct_t *LUstruct,
                             gridinfo3d_t *grid3d)
{

    
    gridinfo_t *grid = &(grid3d->grid2d);
    int_t iam = grid->iam;
    int_t Pc = grid->npcol;
    int_t Pr = grid->nprow;
    int_t myrow = MYROW(iam, grid);
    int_t mycol = MYCOL(iam, grid);

    int_t *xsup = LUstruct->Glu_persist->xsup;
    int_t **Lrowind_bc_ptr = LUstruct->Llu->Lrowind_bc_ptr;
    int_t **Ufstnz_br_ptr = LUstruct->Llu->Ufstnz_br_ptr;
    double **Lnzval_bc_ptr = LUstruct->Llu->Lnzval_bc_ptr;
    double **Unzval_br_ptr = LUstruct->Llu->Unzval_br_ptr;

    lPanelVec = new lpanel_t[CEILING(nsupers, Pc)];
    uPanelVec = new upanel_t[CEILING(nsupers, Pr)];
    // create the lvectors
    for (int_t i = 0; i < CEILING(nsupers, Pc); ++i)
    {
        if (Lrowind_bc_ptr[i] != NULL && isNodeInMyGrid[i * Pc + mycol] == 1)
        {
            lpanel_t lpanel(Lrowind_bc_ptr[i], Lnzval_bc_ptr[i]);
            lPanelVec[i] = lpanel;
        }
    }

    // create the vectors
    for (int_t i = 0; i < CEILING(nsupers, Pr); ++i)
    {
        if (Ufstnz_br_ptr[i] != NULL && isNodeInMyGrid[i * Pr + myrow] == 1)
        {
            upanel_t upanel(Ufstnz_br_ptr[i], Unzval_br_ptr[i]);
            uPanelVec[i] = upanel;
        }
    }

}