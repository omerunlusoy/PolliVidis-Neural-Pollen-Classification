import React, {useEffect} from "react";
import './map.css';
import {
    GoogleMap,
    useLoadScript,
    Marker,
    InfoWindow,
} from "@react-google-maps/api";
import usePlacesAutocomplete, {
    getGeocode,
    getLatLng,
} from "use-places-autocomplete";
import {
    Combobox,
    ComboboxInput,
    ComboboxPopover,
    ComboboxList,
    ComboboxOption,
} from "@reach/combobox";

import "@reach/combobox/styles.css";

//api key: "AIzaSyAHlwtPiz1TdtLSNXtladNYvGRtCbzkm6g"

const libraries = ["places"];
const mapContainerStyle = {
    height: "100vh",
    width: "94vw",
};
const options = {
    //styles: mapStyles,
    disableDefaultUI: true,
    zoomControl: true,
};
const center = {
    lat: 38.749447369341944,
    lng: 34.915717273488404,
};

export default function Map() {
    const { isLoaded, loadError } = useLoadScript({
        googleMapsApiKey: "AIzaSyAHlwtPiz1TdtLSNXtladNYvGRtCbzkm6g",
        libraries,
    });
    const [markers, setMarkers] = React.useState([]);
    const [selected, setSelected] = React.useState(null);

    useEffect(() => {
        fetch(`http://localhost:8000/analysis_posts`)
            .then((data) => data.json())
            .then((data) => setMarkers(data))
    },[]);


    const mapRef = React.useRef();
    const onMapLoad = React.useCallback((map) => {
        mapRef.current = map;
    }, []);

    const panTo = React.useCallback(({ lat, lng }) => {
        mapRef.current.panTo({ lat, lng });
        mapRef.current.setZoom(14);
    }, []);

    if (loadError) return "Error";
    if (!isLoaded) return "Loading...";

    return (
        <div>

            <Locate panTo={panTo} />
            <Search panTo={panTo} />

            <GoogleMap
                id="map"
                mapContainerStyle={mapContainerStyle}
                zoom={6}
                center={center}
                options={options}
                onLoad={onMapLoad}
            >
                {markers.map((marker) => (
                    <Marker
                        key={`${marker.lat}-${marker.lng}`}
                        position={{ lat: marker.lat, lng: marker.lng }}
                        onClick={() => {
                            setSelected(marker);
                        }}
                        icon={{
                            url: `/polen.png`,
                            origin: new window.google.maps.Point(0, 0),
                            anchor: new window.google.maps.Point(15, 15),
                            scaledSize: new window.google.maps.Size(30, 30),
                        }}
                    />
                ))}


                {selected ? (
                    <InfoWindow
                        position={{ lat: selected.lat, lng: selected.lng }}
                        onCloseClick={() => {
                            setSelected(null);
                        }}
                    >
                        <div>
                            <h2>
                                Pollen
                            </h2>
                            <p>Date: {selected.date}</p>
                        </div>
                    </InfoWindow>
                ) : null}
            </GoogleMap>
        </div>
    );
}

function Locate({ panTo }) {
    return (
        <div>
            <button
                className="locate"
                onClick={() => {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            console.log("lat:",position.coords.latitude);
                            console.log("lng:",position.coords.longitude);
                            panTo({
                                lat: position.coords.latitude,
                                lng: position.coords.longitude,
                            });
                        },
                        () => null
                    );

                }}
            >
                <img src="/compass.svg" alt="compass" />
            </button>
        </div>
    );
}

function Search({ panTo }) {
    const {
        ready,
        value,
        suggestions: { status, data },
        setValue,
        clearSuggestions,
    } = usePlacesAutocomplete({
        requestOptions: {
            location: { lat: () => 43.6532, lng: () => -79.3832 },
            radius: 100 * 1000,
        },
    });

    // https://developers.google.com/maps/documentation/javascript/reference/places-autocomplete-service#AutocompletionRequest

    const handleInput = (e) => {
        setValue(e.target.value);
    };

    const handleSelect = async (address) => {
        setValue(address, false);
        clearSuggestions();

        try {
            const results = await getGeocode({ address });
            const { lat, lng } = await getLatLng(results[0]);
            panTo({ lat, lng });
        } catch (error) {
            console.log("😱 Error: ", error);
        }
    };

    return (
        <div className="search">
            <Combobox onSelect={handleSelect}>
                <ComboboxInput
                    value={value}
                    onChange={handleInput}
                    disabled={!ready}
                    placeholder="Search your location"
                />
                <ComboboxPopover>
                    <ComboboxList>
                        {status === "OK" &&
                        data.map(({ id, description }) => (
                            <ComboboxOption key={id} value={description} />
                        ))}
                    </ComboboxList>
                </ComboboxPopover>
            </Combobox>
        </div>
    );
}
