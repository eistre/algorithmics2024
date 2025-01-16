import maplibregl from 'maplibre-gl';
import { useEffect, useRef } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';
import { City, PlanResponse } from '@/types/types';

interface MapProps {
  cities: City[];
  selectedDestinations: City[];
  planResponse: PlanResponse | null;
  startCity: City | null;
}

export function Map({ cities, selectedDestinations, planResponse, startCity }: MapProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<maplibregl.Map | null>(null);

    useEffect(() => {
        if (map.current) return; // initialize map only once
        map.current = new maplibregl.Map({
          container: mapContainer.current!,
          style: {
            version: 8,
            sources: {
              'raster-tiles': {
                type: 'raster',
                tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                tileSize: 256,
                attribution: '<a target="_blank" href="https://openstreetmap.org">© OpenStreetMap</a>',
              }
            },
            layers: [{
              id: 'raster-tiles',
              type: 'raster',
              source: 'raster-tiles',
              minzoom: 0,
              maxzoom: 22
            }]
          },
          // Europe
          center: [17.5, 52.5],
          zoom: 3.25,
        });
      }, []);

    useEffect(() => {
      if (!map.current) return;

      // Clear existing markers
      const markers = document.getElementsByClassName('maplibregl-marker');
      while(markers[0]) {
        markers[0].parentNode?.removeChild(markers[0]);
      }

      // Add markers for all selected destinations
      selectedDestinations.forEach((city) => {
        const coordinates = [city.coordinates[1], city.coordinates[0]] as [number, number];

        // Create a popup
        const popup = new maplibregl.Popup({ offset: 25 })
          .setText(city.name);

        // Add marker with popup
        new maplibregl.Marker({
          color: city === startCity ? 'green' : 'blue',
        })
          .setLngLat(coordinates)
          .setPopup(popup)
          .addTo(map.current!);
      });

    }, [selectedDestinations, startCity]);

    useEffect(() => {
      if (!map.current) return;

      // Make sure to remove existing route
      if (map.current!.getLayer('route')) map.current!.removeLayer('route');
      if (map.current!.getSource('route')) map.current!.removeSource('route');

      if (!planResponse) return;

      const coordinates = planResponse.route.map((city) => {
        const [latitude, longitude] = cities.find((c) => c.name === city)!.coordinates;
        return [longitude, latitude] as [number, number];
      });

      // Draw route on map
      map.current!.addSource('route', {
        type: 'geojson',
        data: {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: coordinates,
          }
        }
      });

      map.current!.addLayer({
        id: 'route',
        type: 'line',
        source: 'route',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': '#888',
          'line-width': 8,
        }
      });

    }, [planResponse, cities]);

    return (<div ref={mapContainer} style={{ width: '100%', minHeight: '85vh' }} />);
}
