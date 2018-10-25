#Tutorial/template for Mapping in Python with Leaflet and Folium
#This file only contains the most useful or common functions for folium, for more in depth go to:
#https://python-visualization.github.io/folium/quickstart.html
#For integration with Dash and Plotly go to:
#https://medium.com/@shachiakyaagba_41915/integrating-folium-with-dash-5338604e7c56
#Import packages
import folium

#Create a base map, simply pass your starting coordinates to Folium
m = folium.Map(location = [45.5236, -122.6570])

#Display in a jupyter notebook just call the variable
m

#Save it in an html file
m.save('index.html')

#Change the default tiles, default is OpenStreetMap
m1 = folium.Map(location = [45.5236, -122.6570], tiles = 'Stamen Toner', zoom_start = 13)
m1.save('index1.html')

#Markers
m2 = folium.Map(location = [45.372, -121.6972], zoom_start = 12, tiles = 'Stamen Terrain')

tooltip = 'Click me!'
folium.Marker([45.3288, -121.6625], popup = '<i>Mt. Hood Meadows</i>', tooltip = tooltip).add_to(m2)
folium.Marker([45.3311, -121.7113], popup = '<b>Timberline Lodge</b>', tooltip = tooltip).add_to(m2)

m2.save('index2.html')

#Change the color and marker icon types (from bootstrap)
m3 = folium.Map(location = [45.372, -121.6972], zoom_start = 12, tiles = 'Stamen Terrain')

folium.Marker([45.3288, -121.6625], popup = 'Mt. Hood Meadows', icon = folium.Icon(icon = 'cloud')).add_to(m3)
folium.Marker([45.3311, -121.7113], popup = 'Timberline Lodge', icon = folium.Icon(color = 'green')).add_to(m3)
folium.Marker([45.3300, -121.6823], popup = 'Some Other Location', icon = folium.Icon(color = 'red', icon = 'info-sign')).add_to(m3)

m3.save('index3.html')

#Add circle markers of various size, see features.py for more options, `Circle` uses meters, `CircleMarker` uses pixels
m4 = folium.Map(location = [45.5236, -122.6570], tiles = 'Stamen Toner', zoom_start = 13)

folium.Circle(radius = 100, location = [45.5244, -122.6699], popup = 'The Waterfront', color = 'crimson', fill = False).add_to(m4)
folium.CircleMarker(location = [45.5215, -122.6261], radius = 50, popup = 'Laurelhurst Park', color = '#3186cc', fill = True, fill_color = '#3186cc').add_to(m4)

m4.save('index4.html')


#A convenience function to enable lat/long popover
m5 = folium.Map(location = [46.1991, -122.1889], tiles = 'Stamen Terrain', zoom_start = 13)

m5.add_child(folium.LatLngPopup())

m5.save('index5.html')

#Click-for-Marker functionality for on-the-fly placement of markers
m6 = folium.Map(location = [46.8527, -121.7649], tiles = 'Stamen Terrain', zoom_start = 13)

folium.Marker([46.8354, -121.7325], popup = 'Camp Muir').add_to(m6)
m6.add_child(folium.ClickForMarker(popup = 'Waypoint'))

m6.save('index6.html')

#GeoJSON and TopoJSON Overlays
washington_school_districts = 'http://geo.wa.gov/datasets/ae00d0d9831544d6a259bd68448546aa_0.geojson'

m7 = folium.Map(location = [47.423316, -120.325279], zoom_start = 5)
folium.GeoJson(washington_school_districts).add_to(m7)

m7.save('index7.html')
