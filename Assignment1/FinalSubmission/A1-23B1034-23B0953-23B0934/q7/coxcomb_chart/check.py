
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference: https://curbal.com/curbal-learning-portal/38-of-100-coxcomb-chart-in-matplotlib
data = {
    "year": [2004, 2022, 2004, 2022, 2004, 2022],
    "countries" : [ "Denmark", "Denmark", "Norway", "Norway","Sweden", "Sweden",],
    "sites": [4,10,5,8,13,15]
}
df= pd.DataFrame(data)

df['year_lbl'] ="'"+df['year'].astype(str).str[-2:].astype(str)
df['sub_total'] = df.groupby('countries')['sites'].transform('sum')

sort_order_dict = {"Denmark":1, "Sweden":2, "Norway":3, 2022:5, 2004:4,}
df = df.sort_values(by=['year','countries',], key=lambda x: x.map(sort_order_dict))
print(df)

countries = df.countries.unique()
years = df.year.unique()
x = len(df.countries.unique())
codes = df.year_lbl
sites = df.sites

# colors = ["#973A36","#4562C5","#141936","#CC5A43","#5475D6","#2C324F",]


fig, ax = plt.subplots(figsize=(5,5),facecolor = "#FFFFFF",subplot_kw=dict(polar=True) )
fig.tight_layout(pad=3.0)

bottom = np.zeros(x)

for year in zip(years,):
    y = df[df["year"] == year]["sites"].values
    x_max = 2*np.pi
    width = x_max/len(countries)
    x_coords = np.linspace(0, x_max, len(countries), endpoint=False)
    ax.bar(x_coords, y,width= width,bottom = bottom,)
    bottom +=y

for bar, site in zip(ax.patches, sites):
    #print(color,site)
    # bar.set_facecolor()
    ax.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height()/2+ bar.get_y(),  #height
        site,
         ha='center', va="center", size=8,
        color = "w", weight= "light",)

ax.set_axis_off()
ax.set_theta_zero_location("N")

plt.show()