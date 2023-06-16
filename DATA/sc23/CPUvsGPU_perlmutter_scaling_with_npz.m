clear all
clc
close all

format shortE

%% read the reference data 

SOLVE_SLU=[];

nrhs = 1;
code = 'superlu_dist_new3Dsolve_nvshmem_03_29_23';
% mats={'s1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin' 'Li4244.bin' 'Ga19As19H42.bin' 'ldoor.mtx'};
% mats={'dielFilterV3real.bin','s1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin'  'Ga19As19H42.bin' 'ldoor.mtx'};
mats={'dielFilterV3real.bin' };


mats_nopost={};
for ii=1:length(mats)
    tmp = mats{ii};
    k = strfind(tmp,'.');
    mats_nopost{1,ii}=tmp(1:k-1);
end


nprows = [1 1 1 1 1 1 1];
npcols = [1 1 1 1 1 1 1];
npzs = [1 2 4 8 16 32 64];
nps = nprows.*npcols.*npzs;




for nm=1:length(mats)
    mat = mats{nm};

    Solve_SLU_CPU=zeros(1,length(nps));
    Solve_SLU_GPU=zeros(1,length(nps));
    Flops_SLU_CPU=zeros(1,length(nps));
    Flops_SLU_GPU=zeros(1,length(nps));
    
    Solve_L_CPU=zeros(1,length(nps));
    Solve_L_GPU=zeros(1,length(nps));
    Solve_U_CPU=zeros(1,length(nps));
    Solve_U_GPU=zeros(1,length(nps));
    Zcomm_CPU=zeros(1,length(nps));
    Zcomm_GPU=zeros(1,length(nps));

    for npp=1:length(nps)
    
        np=nps(npp);
        ncol=npcols(npp);
        nrow=nprows(npp);    
        npz=npzs(npp);   

%         % build is generated with nvhpc and Debug compiling as the Release compiling causes crash in pddistribute. Debug makes CPU solve slower, but doesn't affect GPU solve.   
%         filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve__nrhs_',num2str(nrhs)];

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


        if(nrhs==1)
            filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve_1_nrhs_',num2str(nrhs)];
        else
            filename = ['./',code,'/build_nvidia_nonvshmem/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_gpusolve_1_nrhs_',num2str(nrhs)];
        end
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

%      SOLVE_SLU = [Solve_L_CPU', Solve_L_GPU'];
%  SOLVE_SLU = [Solve_U_CPU', Solve_U_GPU'];

    SOLVE_SLU = zeros(length(nps),2,3);

    SOLVE_SLU(:,1,3) = Solve_L_CPU(:);
    SOLVE_SLU(:,2,3) = Solve_L_GPU(:);
    SOLVE_SLU(:,1,2) = Solve_U_CPU(:);
    SOLVE_SLU(:,2,2) = Solve_U_GPU(:);
    SOLVE_SLU(:,1,1) = Zcomm_CPU(:);
    SOLVE_SLU(:,2,1) = Zcomm_GPU(:);


    figure(nm)
    

    xtick={};
    for ii=1:length(nps)
        tmp = ['$P_z$=',num2str(nps(ii))];
        xtick{1,ii}=tmp;
    end

    origin = [200,60];
    fontsize = 14;
    axisticksize = 24;
    markersize = 10;
    LineWidth = 3;
    
    
%     bar(SOLVE_SLU)

    plotBarStackGroups(SOLVE_SLU, xtick,1,axisticksize-4);
    
    
    %bar((SOLVE_SLU))
    
    % for ii=1:3
    % semilogy(SOLVE_SLU(:,ii), 'LineStyle','-','Color',[lineColors(ii+2,:)],'Marker','o','MarkerSize',markersize,'LineWidth',LineWidth);
    % hold on
    % end
    
    
    
    gca = get(gcf,'CurrentAxes');
    set(gca,'XTick',[1:length(SOLVE_SLU(:,1))])
    set(gca,'TickLabelInterpreter','latex')
    xticklabels(xtick)
%     xtickangle(45)
%     xlim([0,length(mats)+1]);
%     ylim([0,2.2]);
    
    legs = {};
    legs{1,1} = ['CPU Z-Comm'];
    legs{1,2} = ['CPU U-Solve'];
    legs{1,3} = ['CPU L-Solve'];
    legs{1,4} = ['GPU Z-Comm'];
    legs{1,5} = ['GPU U-Solve'];
    legs{1,6} = ['GPU L-Solve'];

    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('Time (s)');
    ylabel(str,'interpreter','latex')
    
    title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


    gca=legend(legs,'interpreter','latex','color','none','NumColumns',1);
    
    set(gcf,'Position',[origin,1000,700]);
    
    fig = gcf;
    style = hgexport('factorystyle');
    style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['CPUvsGPU_perlmuter',mats_nopost{nm},'_nrhs_',num2str(nrhs),'.eps'];
    saveas(fig,str,'epsc')

    tmp = sum(SOLVE_SLU(:,:,:),3)
    tmp(:,1)./tmp(:,2)    

end

