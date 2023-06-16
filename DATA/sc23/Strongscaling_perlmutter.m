clear all
clc
close all

format shortE

%% read the reference data 

SOLVE_SLU=[];

nrhs = 1;
code = 'superlu_dist_new3Dsolve_nvshmem_03_29_23';
% mats={'s1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin' 'Li4244.bin' 'Ga19As19H42.bin' 'ldoor.mtx'};
mats={'dielFilterV3real.bin' 's1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin'  'Ga19As19H42.bin' 'ldoor.mtx'};





mats_nopost={};
for ii=1:length(mats)
    tmp = mats{ii};
    k = strfind(tmp,'.');
    mats_nopost{1,ii}=tmp(1:k-1);
end


nprows = [1 2 4];
npcols = [1 1 1];
npzs = [1 2 4 8 16 32 64];


lineColors = line_colors(length(npzs)+1);


for nm=1:length(mats)
    mat = mats{nm};

    SOLVE_SLU = zeros(length(npzs),length(nprows),3);

    for zz=1:length(npzs)
        npz=npzs(zz);   

        nprows_bac=nprows;
        npcol_bac=npcols;
        if(npz==1)
            nprows = [1 2 4 8];
            npcols = [1 1 1 1];
        end
        Solve_SLU_CPU=zeros(1,length(nprows));
        Solve_SLU_GPU=zeros(1,length(nprows));
        Flops_SLU_CPU=zeros(1,length(nprows));
        Flops_SLU_GPU=zeros(1,length(nprows));
        
        Solve_L_CPU=zeros(1,length(nprows));
        Solve_L_GPU=zeros(1,length(nprows));
        Solve_U_CPU=zeros(1,length(nprows));
        Solve_U_GPU=zeros(1,length(nprows));
        Zcomm_CPU=zeros(1,length(nprows));
        Zcomm_GPU=zeros(1,length(nprows));
    
        for npp=1:length(nprows)
       
            ncol=npcols(npp);
            nrow=nprows(npp);    

% 
%             % build is generated with nvhpc and Debug compiling as the Release compiling causes crash in pddistribute. Debug makes CPU solve slower, but doesn't affect GPU solve.   
%             filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve__nrhs_',num2str(nrhs)];

            % build_gcc is generated with gcc and Release compiling  
            filename = ['./',code,'/build_gcc/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve__nrhs_',num2str(nrhs)];


            fid = fopen(filename);
            while(~feof(fid))
                str=fscanf(fid,'%s',1);
            
                if(strcmp(str,'|forwardSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_CPU(npp)=str;
                end
    
                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_CPU(npp)=str;
                end
    
                if(strcmp(str,'|trs_comm_z'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Zcomm_CPU(npp)=str;
                end
    
                if(strcmp(str,'SOLVE'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'time'))
                       str=fscanf(fid,'%f',1);
                       Solve_SLU_CPU(npp)=str;
                   end
                end    
                if(strcmp(str,'Solve'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'flops'))
                       str=fscanf(fid,'%f',1);
                       Flops_SLU_CPU(npp)=str;
                   end
                end  
            end 
            fclose(fid);



            filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve_1_nrhs_',num2str(nrhs)];
            fid = fopen(filename);
            while(~feof(fid))
                str=fscanf(fid,'%s',1);
            
                if(strcmp(str,'|forwardSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_GPU(npp)=str;
                end
    
                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_GPU(npp)=str;
                end
    
                if(strcmp(str,'|trs_comm_z'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Zcomm_GPU(npp)=str;
                end
    
                if(strcmp(str,'SOLVE'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'time'))
                       str=fscanf(fid,'%f',1);
                       Solve_SLU_GPU(npp)=str;
                   end
                end    
                if(strcmp(str,'Solve'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'flops'))
                       str=fscanf(fid,'%f',1);
                       Flops_SLU_GPU(npp)=str;
                   end
                end  
            end 
            fclose(fid);
 
        end
    
        if(npz>1)
            SOLVE_SLU(zz,:,1)=Solve_SLU_CPU;
            SOLVE_SLU(zz,:,2)=Solve_SLU_GPU;
            SOLVE_SLU(zz,:,3)=npz.*npcols.*nprows;            
        else
            SOLVE_SLU_TMP=zeros(1,length(nprows),3);
            SOLVE_SLU_TMP(1,:,1)=Solve_SLU_CPU;
            SOLVE_SLU_TMP(1,:,2)=Solve_SLU_GPU;
            SOLVE_SLU_TMP(1,:,3)=npz.*npcols.*nprows;
        end

        nprows=nprows_bac;
        npcols=npcol_bac;


    end


    figure(nm)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    
    

    
for ii=1:length(npzs)
npz=npzs(ii);
if(npz==1)
loglog(SOLVE_SLU_TMP(1,:,3), SOLVE_SLU_TMP(1,:,1), 'LineStyle','--','Color',[lineColors(ii,:)],'Marker','v','MarkerSize',markersize,'LineWidth',LineWidth);
else
loglog(SOLVE_SLU(ii,:,3), SOLVE_SLU(ii,:,1), 'LineStyle','--','Color',[lineColors(ii,:)],'Marker','v','MarkerSize',markersize,'LineWidth',LineWidth);
end
hold on
end


for ii=1:length(npzs)
npz=npzs(ii);
if(npz==1)
loglog(SOLVE_SLU_TMP(1,:,3), SOLVE_SLU_TMP(1,:,2), 'LineStyle','-','Color',[lineColors(ii,:)],'Marker','s','MarkerSize',markersize,'LineWidth',LineWidth);
else
loglog(SOLVE_SLU(ii,:,3), SOLVE_SLU(ii,:,2), 'LineStyle','-','Color',[lineColors(ii,:)],'Marker','s','MarkerSize',markersize,'LineWidth',LineWidth);
end
hold on
end



%     bar(SOLVE_SLU)

%     plotBarStackGroups(SOLVE_SLU, xtick);
%     
%     
    %bar((SOLVE_SLU))
    
    % for ii=1:3
    % semilogy(SOLVE_SLU(:,ii), 'LineStyle','-','Color',[lineColors(ii+2,:)],'Marker','o','MarkerSize',markersize,'LineWidth',LineWidth);
    % hold on
    % end
    
    
    
    gca = get(gcf,'CurrentAxes');
    set(gca,'XTick',2.^[0:log2(SOLVE_SLU(end,end,3))])
    set(gca,'TickLabelInterpreter','latex')
%     xticklabels(xtick)
%     xtickangle(45)
     xlim([0,SOLVE_SLU(end,end,3)*1.1]);
%     ylim([0,2.2]);
    
    legs = {};
    legs{1,1} = ['$P_z$=1 CPU'];
    legs{1,2} = ['$P_z$=2 CPU'];
    legs{1,3} = ['$P_z$=4 CPU'];
    legs{1,4} = ['$P_z$=8 CPU'];
    legs{1,5} = ['$P_z$=16 CPU'];
    legs{1,6} = ['$P_z$=32 CPU'];
    legs{1,7} = ['$P_z$=64 CPU'];
    legs{1,8} = ['$P_z$=1 GPU'];
    legs{1,9} = ['$P_z$=2 GPU'];
    legs{1,10} = ['$P_z$=4 GPU'];
    legs{1,11} = ['$P_z$=8 GPU'];
    legs{1,12} = ['$P_z$=16 GPU'];
    legs{1,13} = ['$P_z$=32 GPU'];
    legs{1,14} = ['$P_z$=64 GPU'];

    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('Time (s)');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times P_z$','interpreter','latex')
    
    title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


    gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,1000,700]);
    
    fig = gcf;
    style = hgexport('factorystyle');
    style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['StrongScalingPerlmutter',mats_nopost{nm},'_nrhs_',num2str(nrhs),'.eps'];
    saveas(fig,str,'epsc')

end

