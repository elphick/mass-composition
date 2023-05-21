# MassComposition

[![Run Tests](https://github.com/Elphick/mass-composition/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Elphick/mass-composition/actions/workflows/build_and_test.yml)
[![Publish Docs](https://github.com/Elphick/mass-composition/actions/workflows/docs_to_gh_pages.yml/badge.svg?branch=main)](https://github.com/Elphick/mass-composition/actions/workflows/docs_to_gh_pages.yml)

MassComposition is a python package that allows Geo-scientists and Metallurgists to easily work with, and visualise
mass-compositional data.

Mass Composition in this context is just that - Mass *and* Composition.  It is not Compositional Data Analysis (CoDa), 
since here we care more about mathematical operations like weight averaging and technically correct mathematical 
operations.  Of course, we also need to manage moisture correctly - chemical composition is reported on a dry basis 
and moisture on a wet basis.

The package not only supports individual MassComposition objects, but collections of objects that are 
mathematically related in a Directional Graph (Network or flowsheet).

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of the mass-composition python package.
* You have a Windows/Linux/Mac machine.
* You have read the [docs](https://elphick.github.io/mass-composition).

## Installing MassComposition

To install MassComposition, follow these steps:

```
pip install mass-composition -e .[viz,network]
```

Or, if poetry is more your flavour.

```
poetry add "mass-composition[viz,network]"
```

## Using MassComposition

To use MassComposition, follow these steps:

There are some basic requirements that the incoming DataFrame must meet.  We'll use a sample DataFrame here.

```python    
df_data: pd.DataFrame = sample_data()
```

Create the object

```python
obj_mc: MassComposition = MassComposition(df_data)
```

It is then trivial to calculate the weight average aggregate of the dataset.

```python
obj_mc.aggregate()
```

Multiple composition analytes can be viewed in a single interactive parallel coordinates plot.

```python
obj_mc: MassComposition = MassComposition(df_data.reset_index().set_index(['DHID', 'interval_from', 'interval_to']),
                                          name=name)

fig: Figure = obj_mc.plot_parallel(color='Fe')
fig
```



Network visualisations are and other plots are interactive.

<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="be41dcb6-ad9e-49d4-8427-f08b40237af7" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("be41dcb6-ad9e-49d4-8427-f08b40237af7")) {                    Plotly.newPlot(                        "be41dcb6-ad9e-49d4-8427-f08b40237af7",                        [{"cells":{"align":"left","fill":{"color":[["whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke","whitesmoke","lightgray","whitesmoke"]]},"format":["%s",".1f",".1f",".2f",".2f",".2f",".2f"],"values":[["head","lump","fines"],[2222.598,378.12800000000004,1844.4699999999998],[60.08817446969717,59.76709738501247,60.15399730003742],[0.04553167059450247,0.047710642956882314,0.045084967497438296],[3.1826459305731403,3.527633446875133,3.111921310728828],[1.927222511673276,1.9228835209241315,1.9281120321826872],[8.13386421656098,8.158056107984596,8.12890472601886]]},"columnwidth":[2,1,1,1,1,1,1],"domain":{"x":[0.0,0.45],"y":[0.0,1.0]},"header":{"align":"center","fill":{"color":"cornflowerblue"},"font":{"color":"black","size":12},"values":["name","mass_dry","Fe","P","SiO2","Al2O3","LOI"]},"type":"table"},{"domain":{"x":[0.55,1.0],"y":[0.0,1.0]},"link":{"color":["rgba(143, 90, 57, 255)","rgba(193, 122, 78, 255)","rgba(133, 84, 53, 255)"],"customdata":["<br />mass_wet: 2222.6<br />mass_dry: 2222.6<br />H2O: 0.0<br />Fe: 60.09<br />P: 0.05<br />SiO2: 3.18<br />Al2O3: 1.93<br />LOI: 8.13<br />","<br />mass_wet: 378.13<br />mass_dry: 378.13<br />H2O: 0.0<br />Fe: 59.77<br />P: 0.05<br />SiO2: 3.53<br />Al2O3: 1.92<br />LOI: 8.16<br />","<br />mass_wet: 1844.47<br />mass_dry: 1844.47<br />H2O: 0.0<br />Fe: 60.15<br />P: 0.05<br />SiO2: 3.11<br />Al2O3: 1.93<br />LOI: 8.13<br />"],"hovertemplate":"<b><i>%{label}</i></b><br />Source: %{source.customdata}<br />Target: %{target.customdata}<br />%{customdata}","label":["head","lump","fines"],"source":[0,1,1],"target":[1,2,3],"value":[2222.598,378.12800000000004,1844.4699999999998]},"node":{"color":["blue","green","blue","blue"],"customdata":["0","1","2","3"],"label":["0","1","2","3"],"line":{"color":"black","width":0.5},"pad":15,"thickness":20},"type":"sankey"}],                        {"font":{"size":12},"template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Flowsheet<br>Balanced: True"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>

For full examples, see the [gallery](/auto_examples/index).

## Contributing to MassComposition

To contribute to MassComposition, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin mass-composition`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

This project is still in its infancy, but I'm keen to work with contributors.

## Contact

If you want to contact me you can reach me at <your_email@address.com>.

## License

This project uses the following license: [MIT](/license/license).

