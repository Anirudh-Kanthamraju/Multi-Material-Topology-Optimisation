 %%%%% FINAL INPUT DATA FILE %%%%%
% Defines the upper and lower values and the increment in each case
%%%%% This code generated the input parameters for the matlab code as well
%%%%% as the future python code imlimentations%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Keep the string lengths constant 
% for example the file name should be CNT and not CANTLIVER 
bnd_cnd = "CNT";
RF_min  = 2.0; RF_max  = 2; RF_inc  = 2;
num_mat = 3; 
% num_mat = 4;
mat_names = 'abc';
mod = [3 2 1];
ratio= [1 2];
p = permutation( num_mat , mat_names, mod, ratio);
% meshx_min  = 100; meshx_incriment = 50; meshx_max= 200;
voidVF_min = 0.2; voidVF_max = 0.5; voidVF_increment= 0.01;     
file_name = '';
serial_number = 1;
data = {};
for asp = [RF_min:RF_inc:RF_max]
    
    
    %for m = [meshx_min : meshx_incriment: meshx_max]
    
        for vf = [voidVF_min : voidVF_increment : voidVF_max]

            for mat = [1:1: height(p)]
                
                temp1 = table2cell(p(mat, 'materials'));
%                 file_name = strcat( bnd_cnd , '_' , sprintf('%1.1f', asp ) ,  '_' , sprintf( '%1.0f', '100*50' ), '_' , temp1{1,1} , '_' , sprintf( '%1.3f', vf ));
                data{serial_number,1}= serial_number;
%                 data{serial_number,2}= file_name;
                data{serial_number,3}= 100;
                data{serial_number,4}= 50;
                temp4 = p(mat, 'materials');
                data{serial_number,5}= cell2mat(temp4{1,1});
                data{serial_number,6}= vf;
                temp2 = p(mat, 'youngs modulus');
                temp3 = p(mat, 'colour_matrix');
                temp4 = p(mat, 'Mat_vol_ratio');
                data{serial_number,7}= cell2mat(temp2{1,1});
                data{serial_number,8}= cell2mat(temp3{1,1});
                data{serial_number,9}=[cell2mat(temp4{1,1})*(1-vf) vf];
                data{serial_number,10}= asp;
                mesh_name='100x50';
                file_name=strcat( bnd_cnd , '_',mesh_name, '_',num2str([cell2mat(temp4{1,1})*(1-vf) vf]));
%                 file_name = strcat( bnd_cnd , '_' ,sprintf( '%1.0f', '100*50' ), '_' , num2str([cell2mat(temp4{1,1})*(1-vf) vf])); 
                data{serial_number,2}= file_name;
                serial_number = serial_number+1;                
            end 
        end 
        
    %end 
end
data = cell2table(data);
data.Properties.VariableNames = {'serial_number ', 'file name', 'Mx', 'My' , 'materials' ,'Void VF' , 'youngs modulus' , 'colour matrix', 'Mat_vol_frac', 'Radius_Filter' };
% writetable ( data , 'cantiliver.csv')

%%%%%%%%%%%%%%%%This section of the code is there to sample uniformally all
%%%%%%%%%%%%%%%%the data that has been geneted in the previous section .
%%%%%%%%%%%%%%%%For the application here under a sample of 200 images will
%%%%%%%%%%%%%%%%be generated for each mesh size
no_sel=250;
total= height(data);
ratio = height(data)/250;
i= 1; 
data_sample = {};
flag=0;
sum=1;
j=1;
while flag==0
    
    data_sample(i,:)=table2cell(data(j,:));
    i=i+1;
    sum=sum+ratio;
    j=floor(sum);
    if j<=height(data)
        flag=0;
    else 
        flag=1;
    end 
end 




for i = [121]
    input = data_sample(i,:)
 

[nx,ny,tol_out,tol_f,iter_max_in,iter_max_out,p,q,e,v,rf,name, clr] = set_parameters (input);
multi_top(nx,ny,tol_out,tol_f,iter_max_in,iter_max_out,p,q,e,v,rf, name, clr);

end 

function [nx,ny,tol,tolf,im_in,im_out,p,q,e,v,rf, name, clr] = set_parameters(input)
% 
mate= cell2mat(input(1,5));
a = length ( mate);
b = count ( mate , 'x') ; 
p = a-b+1 ; %number of materials +1 to count the void 


nx = 200; ny = 100; tol = 0.01; tolf = 0.5; im_in = 2; im_out = 500;
q = 3; 
e = cell2mat(input(1,7))'; v = cell2mat(input(1,9))'; rf = 3.5; 
name=strcat(int2str(cell2mat((input(1,1)))),'.jpg');
clr= cell2mat(input(1,8));
end

function multi_top(nx,ny,tol_out,tol_f,iter_max_in,iter_max_out,p,q,e,v,rf, name, clr)
alpha = zeros(nx*ny,p);
for i = 1:p
alpha(:,i) = v(i);
end
% MAKE FILTER
 [H,Hs] = make_filter (nx,ny,rf);
 change_out = 2*tol_out; iter_out = 0;
 while (iter_out < iter_max_out) && (change_out > tol_out)
 alpha_old = alpha;
 for a = 1:p
 for b = a+1:p
 [obj,alpha] = bi_top(a,b,nx,ny,p,v,e,q,alpha,H,Hs,iter_max_in);
 end
 end
 iter_out = iter_out + 1;
 change_out = norm(alpha(:)-alpha_old(:),inf);
 fprintf('Iter:%5i Obj.:%11.4f change:%10.8f\n',iter_out,obj,change_out);
 % UPDATE FILTER
 if (change_out < tol_f) && (rf>3)
 tol_f = 0.99*tol_f; rf = 0.99*rf; [H,Hs] = make_filter (nx,ny,rf);
 end
 % SCREEN OUT TEMPORAL TOPOLOGY
 I = make_bitmap (p,nx,ny,alpha, clr);
 image(I), axis image off, drawnow;
 if change_out<= tol_out | iter_out==500
     %     k= data.x1(itr);
     
    imwrite ( I ,name)
    
else
    flag=0;
end
 end
end
 
 function [H,Hs] = make_filter (nx,ny,rmin)
 ir = ceil(rmin)-1;
 iH = ones(nx*ny*(2*ir+1)^2,1);
 jH = ones(size(iH));
 sH = zeros(size(iH));
 k = 0;
 for i1 = 1:nx
 for j1 = 1:ny
 e1 = (i1-1)*ny+j1;
 for i2 = max(i1-ir,1):min(i1+ir,nx)
 for j2 = max(j1-ir,1):min(j1+ir,ny)
 e2 = (i2-1)*ny+j2; k = k+1; iH(k) = e1; jH(k) = e2;
 sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
 end
 end
 end
 end
 H = sparse(iH,jH,sH); Hs = sum(H,2);
 end
 
function I = make_bitmap (p,nx,ny,alpha, clr) 
 color = clr;
 I = zeros(nx*ny,3);
 for j = 1:p
 I(:,1:3) = I(:,1:3) + alpha(:,j)*color(j,1:3);
 end
 I = imresize(reshape(I,ny,nx,3),1,'bilinear'); 
end

function [o,alpha] = bi_top(a,b,nx,ny,p,v,e,q,alpha_old,H,Hs,iter_max_in)
 alpha = alpha_old; iter_in = 0; nu = 0.3;
 %% PREPARE FINITE ELEMENT ANALYSIS
 A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12];
 A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6];
 B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4];
 B12 = [ 2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2];
 KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
 nodenrs = reshape(1:(1+nx)*(1+ny),1+ny,1+nx);
 edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nx*ny,1);
 edofMat = repmat(edofVec,1,8)+repmat([0 1 2*ny+[2 3 0 1] -2 -1],nx*ny,1);
 iK = reshape(kron(edofMat,ones(8,1))',64*nx*ny,1);
 jK = reshape(kron(edofMat,ones(1,8))',64*nx*ny,1);
 %% DEFINE LOADS AND SUPPORTS (CANTILEVER BEAM)
 F = sparse(2*(ny+1)*(nx+1),1,-1,2*(ny+1)*(nx+1),1);
 fixeddofs = [1:2*ny+1];
 U = zeros(2*(ny+1)*(nx+1),1);
 alldofs = [1:2*(ny+1)*(nx+1)];
 freedofs = setdiff(alldofs,fixeddofs);
 %% INNER ITERATIONS
 while iter_in < iter_max_in
 iter_in = iter_in + 1;
 %% FE-ANALYSIS
 E = e(1)*alpha(:,1).^q;
 for phase = 2:p
 E = E + e(phase)*alpha(:,phase).^q;
 end
 sK = reshape(KE(:)*E(:)',64*nx*ny,1);
 K = sparse(iK,jK,sK); K = (K+K')/2;
 U(freedofs) = K(freedofs,freedofs)\F(freedofs);
 %% OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
 ce = sum((U(edofMat)*KE).*U(edofMat),2);
 o = sum(sum(E.*ce));
 dc = -(q*(e(a)-e(b))*alpha(:,a).^(q-1)).*ce;
 %% FILTERING OF SENSITIVITIES
 dc = H*(alpha(:,a).*dc)./Hs./max(1e-3,alpha(:,a)); dc = min(dc,0);
 %% UPDATE LOWER AND UPPER BOUNDS OF DESIGN VARIABLES
 move = 0.2;
 r = ones(nx*ny,1);
 for k = 1:p
 if (k ~= a) && (k ~= b)
 r = r - alpha(:,k);
 end
end
 l = max(0,alpha(:,a)-move);
 u = min(r,alpha(:,a)+move);
%% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES
 l1 = 0; l2 = 1e9;
 while (l2-l1)/(l1+l2) > 1e-3
 lmid = 0.5*(l2+l1);
 alpha_a = max(l,min(u,alpha(:,a).*sqrt(-dc./lmid)));
 if sum(alpha_a) > nx*ny*v(a); l1 = lmid; else l2 = lmid; end
 end
 alpha(:,a) = alpha_a;
 alpha(:,b) = r-alpha_a;
 end
end

function [mat_data_table_fin] = permutation (num_mat, mat_names, mod, ratio)
    clr1 = [1 0 0];
    clr2 = [0 1 0];
    clr3 = [0 0 1];                                                        
    clr4 = [0.5 0.5 0];
    clr5 = [0.5 0 0.5];
    void = [1 1 1];
    clr = {clr1,clr2,clr3,clr4,clr5,void};                                 
    keys = {};
    val_col = {};
    mat_data_temp = {};
    for i = [1:1:num_mat]
        keys{i,1}= mat_names(i);           
        val_col{i,1}= clr(i);   % just creating a matrix with individual colours and materials to be used later. 
    end
    mat_mod= containers.Map(keys , mod); % associationg each material to a colour.
    mat_col= containers.Map(keys ,val_col);% associationg each material to a modulus.
    temp = '';
    for  i=[1:1:num_mat]
        temp = strcat(temp , 'x');
    end
    iter = 1;
    for  k=[1:1:num_mat]
        comb = nchoosek(mat_names,k);
        str = temp(1:length(temp)-k);
        if k == num_mat
            length_comb = 1;
        else 
            length_comb = length (comb);
        end 
        for j = [1:1: length_comb];
            MAT = strcat(str, comb(j,:));
%             mat_data{iter,1} = iter; 
            mat_data_temp{iter,2} = MAT;
            modulus = []; 
            color = [];
            mat_choice = comb(j,:);
            for l = [1:1:length(mat_choice)]
                mat = mat_choice(l);
                modulus(1,l) = cell2mat(values(mat_mod,  {mat}));
                var = values(mat_col, {mat});
                color(l,:)= cell2mat(var{1,1}(1,1));
                    
            end 
            modulus (1,length(mat_choice)+1) = 1e-9;
            color(length(mat_choice)+1,:)= void;
            mat_data_temp{iter,3} = modulus;
            mat_data_temp{iter,4} = color;
            iter = iter + 1;
        end               
    end
    mat_data_table = cell2table (mat_data_temp);
    
    mat_data = {};
    h = height (mat_data_table);
    itr =  1;
    for i = [1:1:h] 
           mat = mat_data_temp {i, 2};
           a = length ( mat);
           b = count ( mat , 'x') ;
           num = a - b ;
               
    if num == 1 
         mat_frac= [1];
         mat_data{itr, 1} = itr; 
         mat_data{itr, 2} = mat_data_temp {i, 2};
         mat_data{itr, 3} = mat_data_temp {i, 3};
         mat_data{itr, 4} = mat_data_temp {i, 4};
         mat_data{itr, 5} = mat_frac;
         itr = itr +1; 
    elseif num == 2 
        list_2 = [];
        for j = [1:1: length( ratio )]
            for k = [1:1: length( ratio )]
                    sum = ratio(j) +  ratio(k);
                    mat = ([ratio(j) ratio(k)]/sum) ;
                    list_2 = [list_2 ; mat];
                    list_2 = unique(list_2 , 'rows');
            end
        end
        for x = [1:1:length(list_2)]
             mat_frac= list_2(x,:);
             mat_data{itr, 1} = itr; 
             mat_data{itr, 2} = mat_data_temp {i, 2};
             mat_data{itr, 3} = mat_data_temp {i, 3};
             mat_data{itr, 4} = mat_data_temp {i, 4};
             mat_data{itr, 5} = mat_frac;
             itr = itr +1;
        end     
    else
        list_3 = [];
        for j = [1:1: length( ratio )]
            for k = [1:1: length( ratio )]
                for l = [1:1: length( ratio )]
                    sum = ratio(j) +  ratio(k) + ratio (l); 
                    mat = ([ratio(j) ratio(k) ratio(l) ]/sum) ;
                    list_3 = [list_3 ; mat];
                    list_3 = unique(list_3 , 'rows');
                end 
            end 
        end
        for x = [1:1:length(list_3)]
             mat_frac= list_3(x,:);
             mat_data{itr, 1} = itr; 
             mat_data{itr, 2} = mat_data_temp {i, 2};
             mat_data{itr, 3} = mat_data_temp {i, 3};
             mat_data{itr, 4} = mat_data_temp {i, 4};
             mat_data{itr, 5} = mat_frac;
             itr = itr +1;
        end     
    end
    end 
  mat_data_table_fin = cell2table (mat_data);  
  mat_data_table_fin.Properties.VariableNames = {'serial_number ', 'materials' , 'youngs modulus' , 'colour_matrix', 'Mat_vol_ratio' };  
end


 
 
 