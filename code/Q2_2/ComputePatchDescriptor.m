function [ descriptor ] = ComputePatchDescriptor( image, gradient_bins )
    
    shape = size(image);
    descriptor = zeros(1, gradient_bins);
    [Gmag, Gdir] = imgradient(image);
    Gdir = abs(Gdir);
    
    for i = 1:shape(1)
        for j= 1:shape(2)
            weight2 = rem(Gdir(i,j), 20)/20;
            weight1 = 1-weight2;
            bin_no = ceil(Gdir(i,j)/20);
            if (bin_no == 0)
                bin_no = 1;
            end
            bin_no2 = bin_no+1;
            if (bin_no2 == 10)
                bin_no2 = 1;
            end
            descriptor(1, bin_no) = descriptor(1, bin_no) + (weight1*Gmag(i,j));
            descriptor(1, bin_no2) = descriptor(1, bin_no2) + (weight2*Gmag(i,j));
        end
    end
    
end