clear all
clc
close all

format shortE

%% read the reference data 

SOLVE_SLU=[];

cmap='parula';
% cmap='hot';

nrhs = 1;
code = 'superlu_dist_new3Dsolve_03_25_23';
% mats={'s1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin' 'Li4324.bin' 'Ga19As19H42.bin' 'ldoor.mtx'};
% mats={'s2D9pt2048.rua'};

mats={'nlpkkt80.bin'};




mats_nopost={};
for ii=1:length(mats)
    tmp = mats{ii};
    k = strfind(tmp,'.');
    mats_nopost{1,ii}=tmp(1:k-1);
end


nprows = [1 2 4];
npcols = [1 1 1];
% 
% npzs = [1 2 4 8 16 32 64];
% nprows_all = [8 8 4 4 2 2 1; 16 8 8 4 4 2 2; 16 16 8 8 4 4 2; 32 16 16 8 8 4 4; 64 32 16 16 8 8 4];
% npcols_all = [16 8 8 4 4 2 2; 16 16 8 8 4 2 2; 32 16 16 8 8 4 4; 32 32 16 16 8 8 4; 32 32 32 16 16 8 8];


npzs = [1 2 4 8 16 32 ];
nprows_all = [8 8 4 4 2 2 ; 16 8 8 4 4 2 ; 16 16 8 8 4 4 ; 32 16 16 8 8 4 ; 64 32 16 16 8 8 ];
npcols_all = [16 8 8 4 4 2 ; 16 16 8 8 4 2 ; 32 16 16 8 8 4 ; 32 32 16 16 8 8 ; 32 32 32 16 16 8 ];
lineColors = line_colors(length(npzs)+1);


for nm=1:length(mats)
    mat = mats{nm};

    SOLVE_SLU = zeros(length(npzs),length(nprows_all(:,1)),3);
    SOLVE_COMM_Z_OLD = zeros(length(npzs),length(nprows_all(:,1)));
    SOLVE_COMP_2D_OLD = zeros(length(npzs),length(nprows_all(:,1)));
    SOLVE_COMM_2D_OLD = zeros(length(npzs),length(nprows_all(:,1)));

    SOLVE_COMM_Z_NEWEST = zeros(length(npzs),length(nprows_all(:,1)));
    SOLVE_COMP_2D_NEWEST = zeros(length(npzs),length(nprows_all(:,1)));
    SOLVE_COMM_2D_NEWEST = zeros(length(npzs),length(nprows_all(:,1)));


    for zz=1:length(npzs)
        npz=npzs(zz);  
        nprows=nprows_all(:,zz);
        npcols=npcols_all(:,zz);

        Solve_SLU_OLD=zeros(1,length(nprows));
        Solve_SLU_NEWEST=zeros(1,length(nprows));
        Flops_SLU_OLD=zeros(1,length(nprows));
        Flops_SLU_NEWEST=zeros(1,length(nprows));
        
        Solve_L_OLD=zeros(1,length(nprows));
        Solve_L_COMP_OLD=zeros(1,length(nprows));
        Solve_L_COMM_OLD=zeros(1,length(nprows));
        Solve_L_NEWEST=zeros(1,length(nprows));
        Solve_L_NEWEST_OLD=zeros(1,length(nprows));
        Solve_L_NEWEST_OLD=zeros(1,length(nprows));

        Solve_U_OLD=zeros(1,length(nprows));
        Solve_U_COMP_OLD=zeros(1,length(nprows));
        Solve_U_COMM_OLD=zeros(1,length(nprows));        
        Solve_U_NEWEST=zeros(1,length(nprows));
        Solve_U_COMP_NEWEST=zeros(1,length(nprows));
        Solve_U_COMM_NEWEST=zeros(1,length(nprows));          
        Zcomm_OLD=zeros(1,length(nprows));
        Zcomm_NEWEST=zeros(1,length(nprows));
    
        for npp=1:length(nprows)
       
            ncol=npcols(npp);
            nrow=nprows(npp);    



            filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_old_nrhs_',num2str(nrhs)];
            fid = fopen(filename);
            while(~feof(fid))
                str=fscanf(fid,'%s',1);
            
                if(strcmp(str,'|forwardSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_OLD(npp)=str;
                end

                if(strcmp(str,'|forwardSolve-compute'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_COMP_OLD(npp)=str;
                end

                if(strcmp(str,'|forwardSolve-comm'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_COMM_OLD(npp)=str;
                end                

                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_OLD(npp)=str;
                end
    
                if(strcmp(str,'|backSolve-compute'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_COMP_OLD(npp)=str;
                end

                if(strcmp(str,'|backSolve-comm'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_COMM_OLD(npp)=str;
                end                


                if(strcmp(str,'|trs_comm_z'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Zcomm_OLD(npp)=str;
                end
    
                if(strcmp(str,'SOLVE'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'time'))
                       str=fscanf(fid,'%f',1);
                       Solve_SLU_OLD(npp)=str;
                   end
                end    
                if(strcmp(str,'Solve'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'flops'))
                       str=fscanf(fid,'%f',1);
                       Flops_SLU_OLD(npp)=str;
                   end
                end  
            end 
            fclose(fid);



            filename = ['./',code,'/build/',mat,'/SLU.o_mpi_',num2str(nrow),'x',num2str(ncol),'x',num2str(npz),'_1_3d_newest_nrhs_',num2str(nrhs)];
            fid = fopen(filename);
            while(~feof(fid))
                str=fscanf(fid,'%s',1);
            
                if(strcmp(str,'|forwardSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_NEWEST(npp)=str;
                end
    

                if(strcmp(str,'|forwardSolve-compute'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_COMP_NEWEST(npp)=str;
                end

                if(strcmp(str,'|forwardSolve-comm'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_L_COMM_NEWEST(npp)=str;
                end                


                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_NEWEST(npp)=str;
                end
    
    
                if(strcmp(str,'|backSolve-compute'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_COMP_NEWEST(npp)=str;
                end

                if(strcmp(str,'|backSolve-comm'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_COMM_NEWEST(npp)=str;
                end     

                if(strcmp(str,'|trs_comm_z'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Zcomm_NEWEST(npp)=str;
                end
    
                if(strcmp(str,'SOLVE'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'time'))
                       str=fscanf(fid,'%f',1);
                       Solve_SLU_NEWEST(npp)=str;
                   end
                end    
                if(strcmp(str,'Solve'))
                   str=fscanf(fid,'%s',1);
                   if(strcmp(str,'flops'))
                       str=fscanf(fid,'%f',1);
                       Flops_SLU_NEWEST(npp)=str;
                   end
                end  
            end 
            fclose(fid);
    
    
        end
    
    
        SOLVE_SLU(zz,:,1)=Solve_SLU_OLD;
        SOLVE_SLU(zz,:,2)=Solve_SLU_NEWEST;
        SOLVE_SLU(zz,:,3)=npz.*npcols.*nprows;

        SOLVE_COMM_Z_OLD(zz,:)=Zcomm_OLD;
        SOLVE_COMP_2D_OLD(zz,:)=Solve_L_COMP_OLD+Solve_U_COMP_OLD;
        SOLVE_COMM_2D_OLD(zz,:)=Solve_L_COMM_OLD+Solve_U_COMM_OLD;


        SOLVE_COMM_Z_NEWEST(zz,:)=Zcomm_NEWEST;
        SOLVE_COMP_2D_NEWEST(zz,:)=Solve_L_COMP_OLD+Solve_U_COMP_NEWEST;
        SOLVE_COMM_2D_NEWEST(zz,:)=Solve_L_COMM_OLD+Solve_U_COMM_NEWEST;

    end


    figure(1)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMM_Z_OLD);

    gca = get(gcf,'CurrentAxes');
     
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);
 
 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMM_Z_OLD,SOLVE_COMM_Z_NEWEST))) max(max(max(SOLVE_COMM_Z_OLD,SOLVE_COMM_Z_NEWEST)))]);
    set(gca,'FontSize',32) 
%     axis equal;
    shading interp;

    title(['Z-Comm (baseline)'],'interpreter','none');

    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times Pz$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_z_comm',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_old.eps'];
    saveas(fig,str,'epsc')


    figure(2)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMM_Z_NEWEST);
    gca = get(gcf,'CurrentAxes');

    
%     set(gca,'XTick',0:1:ncol)
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);

 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMM_Z_OLD,SOLVE_COMM_Z_NEWEST))) max(max(max(SOLVE_COMM_Z_OLD,SOLVE_COMM_Z_NEWEST)))]);
    set(gca,'FontSize',32) 
%     axis equal;
    shading interp;
    title(['Z-Comm (proposed)'],'interpreter','none');


    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times Pz$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_z_comm',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_new.eps'];
    saveas(fig,str,'epsc')





    figure(3)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMM_2D_OLD);

    gca = get(gcf,'CurrentAxes');
     
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);

 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMM_2D_OLD,SOLVE_COMM_2D_NEWEST))) max(max(max(SOLVE_COMM_2D_OLD,SOLVE_COMM_2D_NEWEST)))]);
    set(gca,'FontSize',32) 
%     axis equal;
    shading interp;
    title(['XY-Comm (baseline)'],'interpreter','none');



    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times Pz$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_xy_comm',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_old.eps'];
    saveas(fig,str,'epsc')

    figure(4)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMM_2D_NEWEST);
    gca = get(gcf,'CurrentAxes');

    
%     set(gca,'XTick',0:1:ncol)
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);

 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMM_2D_OLD,SOLVE_COMM_2D_NEWEST))) max(max(max(SOLVE_COMM_2D_OLD,SOLVE_COMM_2D_NEWEST)))]);
    set(gca,'FontSize',32) 
%     axis equal;
    shading interp;

    title(['XY-Comm (proposed)'],'interpreter','none');

    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times Pz$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_xy_comm',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_new.eps'];
    saveas(fig,str,'epsc')





    figure(5)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMP_2D_OLD);

    gca = get(gcf,'CurrentAxes');
     
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);

 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMP_2D_OLD,SOLVE_COMP_2D_NEWEST))) max(max(max(SOLVE_COMP_2D_OLD,SOLVE_COMP_2D_NEWEST)))]);
    set(gca,'FontSize',32) 
%     axis equal;
    shading interp;
    title(['FP-Operation (baseline)'],'interpreter','none');


    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times P_z$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_xy_compute',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_old.eps'];
    saveas(fig,str,'epsc')


    figure(6)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    colormap(cmap);
    
    imagesc(SOLVE_COMP_2D_NEWEST);
    gca = get(gcf,'CurrentAxes');

    
%     set(gca,'XTick',0:1:ncol)
%     set(gca,'YTick',npzs)
    Ylabels={};
    for ii=1:length(npzs)
        Ylabels{ii}=num2str(npzs(ii));
    end
    yticklabels(Ylabels);

    nprocxy=nprows_all.*npcols_all;
    nmpis=nprocxy(:,1);
    Xlabels={};
    for ii=1:length(nmpis)
        Xlabels{ii}=num2str(nmpis(ii));
    end
    set(gca,'XTick',1:1:length(nmpis));
    xticklabels(Xlabels);

 
    shading flat
    cax_loc = 'eastoutside';
    % c = colorbar('location','North');
    c = colorbar('location',cax_loc);
    caxis([min(min(min(SOLVE_COMP_2D_OLD,SOLVE_COMP_2D_NEWEST))) max(max(max(SOLVE_COMP_2D_OLD,SOLVE_COMP_2D_NEWEST)))]);
    set(gca,'FontSize',32) 
%      axis equal;
    shading interp;

    title(['FP-Operation (proposed)'],'interpreter','none');

    gca = get(gcf,'CurrentAxes');
    set( gca, 'FontName','Times New Roman','fontsize',axisticksize);
    str = sprintf('$P_z$');
    ylabel(str,'interpreter','latex')
    xlabel('$P_x\times P_y\times Pz$','interpreter','latex')
    
%     title([mats_nopost{nm},' nrhs=',num2str(nrhs)],'interpreter','none');


%     gca=legend(legs,'interpreter','latex','color','none','NumColumns',2);
    
    set(gcf,'Position',[origin,700,660]);
    
    fig = gcf;
%     style = hgexport('factorystyle');
%     style.Bounds = 'tight';
    % hgexport(fig,'-clipboard',style,'applystyle', true);
    
    
    str = ['Profiling_xy_compute',mats_nopost{nm},'_nrhs_',num2str(nrhs),'_new.eps'];
    saveas(fig,str,'epsc')


end

