Troubleshooting
===============

You may be having trouble if you landed here, or are looking for tips-n-tricks to keep you out of trouble ;-).

Why won't my plotly plot show?
------------------------------

If you have followed the example in your IDE, you may have code like below and be wondering where your plot is.

..  code-block:: python

    from plotly.graph_objs import Figure
    fig: Figure = obj_mc.plot_parallel()
    fig

The examples are documented using Sphinx, which behaves like Jupyter Notebooks.

To get the plot to show using your IDE you need this simple tweak and invoke the show method.

..  code-block:: python

    from plotly.graph_objs import Figure
    fig: Figure = obj_mc.plot_parallel()
    fig.show()


Why did I just run out of memory?
---------------------------------

Did you inadvertently index your incoming DataFrame with too many dimensions?
It can be surprising how quickly your RAM can be consumed by xarray when using multiple dimensions.

Imagine you have a DataFrame with 1000 records and an integer index, you'll make a MassComposition object no problem.
But if you then set multiple indexes, say 3, you will consume more memory.  Why?

Xarray will densify the Dataset, which means it will place nans in positions where there are no values in N-D space,
in this case 5-D.  While I'm not certain, this may result in 1000^3 = 1,000,000,000 records with 9,999,999,000 nans.
I could be wrong on the specifics - I should check, but if you have memory issues with many Xarray dimensions
(a.k.a many multiple pandas indexes on your incoming DataFrame), then simply reset the DataFrame index to get back to
1-D and go from there.

YMMV - subject to your data and what you're trying to achieve.