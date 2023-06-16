clear all
clc
close all

format shortE

%% read the reference data 

SOLVE_SLU=[];

nrhs = 1;
code = 'superlu_dist_new3Dsolve_03_25_23';
% mats={'s1_mat_0_253872.bin' 's2D9pt2048.rua' 'nlpkkt80.bin' 'Li4244.bin' 'Ga19As19H42.bin' 'ldoor.mtx'};
% mats={'s2D9pt2048.rua' 'nlpkkt80.bin' 'ldoor.bin' 'dielFilterV3real.bin'};

% 
% mats={'s2D9pt2048.rua'};
% npzs = [1 4  16  ];
% nprows_all = [8  4  2  ; 16  8  4  ; 16  8  4  ; 32  16  8  ; 64  16  8  ];
% npcols_all = [16  8  4  ; 16  8  4  ; 32  16  8  ; 32  16  8  ; 32  32  16  ];
 

% mats={'ldoor.bin'};
% npzs = [1 4  16  ];
% nprows_all = [8  4  2  ; 16  8  4  ; 16  8  4  ; 32  16  8  ; 64  16  8  ];
% npcols_all = [16  8  4  ; 16  8  4  ; 32  16  8  ; 32  16  8  ; 32  32  16  ];
 

% mats={'nlpkkt80.bin'};
% npzs = [1 4  16  ];
% nprows_all = [8  4  2  ; 16  8  4  ; 16  8  4  ; 32  16  8  ; 64  16  8  ];
% npcols_all = [16  8  4  ; 16  8  4  ; 32  16  8  ; 32  16  8  ; 32  32  16  ];
%  


mats={'dielFilterV3real.bin'};
npzs = [1 4  16  ];
nprows_all = [8  4  2  ; 16  8  4  ; 16  8  4  ; 32  16  8   ];
npcols_all = [16  8  4  ; 16  8  4  ; 32  16  8  ; 32  16  8   ];
 

mats_nopost={};
for ii=1:length(mats)
    tmp = mats{ii};
    k = strfind(tmp,'.');
    mats_nopost{1,ii}=tmp(1:k-1);
end


nprows = [1 2 4];
npcols = [1 1 1];


lineColors = line_colors(length(npzs)+1);


for nm=1:length(mats)
    mat = mats{nm};

    SOLVE_SLU = zeros(length(npzs),length(nprows_all(:,1)),3);

    for zz=1:length(npzs)
        npz=npzs(zz);  
        nprows=nprows_all(:,zz);
        npcols=npcols_all(:,zz);

        Solve_SLU_OLD=zeros(1,length(nprows));
        Solve_SLU_NEWEST=zeros(1,length(nprows));
        Flops_SLU_OLD=zeros(1,length(nprows));
        Flops_SLU_NEWEST=zeros(1,length(nprows));
        
        Solve_L_OLD=zeros(1,length(nprows));
        Solve_L_NEWEST=zeros(1,length(nprows));
        Solve_U_OLD=zeros(1,length(nprows));
        Solve_U_NEWEST=zeros(1,length(nprows));
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
    
                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_OLD(npp)=str;
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
    
                if(strcmp(str,'|backSolve'))
                    str=fscanf(fid,'%s',1);
                    str=fscanf(fid,'%f',1);
                    Solve_U_NEWEST(npp)=str;
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

    end


    figure(nm)
    
    origin = [200,60];
    fontsize = 32;
    axisticksize = 32;
    markersize = 10;
    LineWidth = 3;
    
    

    
for ii=1:length(npzs)
loglog(SOLVE_SLU(ii,:,3), SOLVE_SLU(ii,:,1), 'LineStyle','--','Color',[lineColors(ii,:)],'Marker','v','MarkerSize',markersize,'LineWidth',LineWidth);
hold on
end


for ii=1:length(npzs)
loglog(SOLVE_SLU(ii,:,3), SOLVE_SLU(ii,:,2), 'LineStyle','-','Color',[lineColors(ii,:)],'Marker','s','MarkerSize',markersize,'LineWidth',LineWidth);
hold on
end

     xlim([0,SOLVE_SLU(end,end,3)*1.1]);

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
%     xlim([0,length(mats)+1]);
%     ylim([0,2.2]);
    
    legs = {};
    legs{1,1} = ['$P_z$=1 Baseline'];
    legs{1,2} = ['$P_z$=4 Baseline'];
    legs{1,3} = ['$P_z$=16 Baseline'];

    legs{1,4} = ['$P_z$=1 New'];
    legs{1,5} = ['$P_z$=4 New'];
    legs{1,6} = ['$P_z$=16 New'];


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
    
    
    str = ['StrongScalingCori',mats_nopost{nm},'_nrhs_',num2str(nrhs),'.eps'];
    saveas(fig,str,'epsc')

end

