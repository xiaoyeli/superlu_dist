function lineColors = line_colors(lineCount)

    %// For more control - move these four lines outside of the function and make replace lineCount as a parameter with lineColors
    t = linspace(0,1,lineCount)';                              %//'
    s = 1/2 + zeros(lineCount,1);
    v = 0.8*ones(lineCount,1);
    lineColors = colormap(squeeze(hsv2rgb(t,s,v)));
    
end