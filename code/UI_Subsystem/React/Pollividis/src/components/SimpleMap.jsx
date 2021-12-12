import React from "react";
import GoogleMapReact from 'google-map-react';

const AnyReactComponent = ({ text }) => <div>{text}</div>;

export default function SimpleMap(){
    const defaultProps = {
        center: {
            lat: 38.749447369341944,
            lng: 34.915717273488404
        },
        zoom: 7
    };

    return (
        // Important! Always set the container height explicitly
        <div style={{ height: '100vh', width: '100%' }}>
            <GoogleMapReact
                bootstrapURLKeys={{ key: "AIzaSyAHlwtPiz1TdtLSNXtladNYvGRtCbzkm6g" }}
                defaultCenter={defaultProps.center}
                defaultZoom={defaultProps.zoom}
                yesIWantToUseGoogleMapApiInternals
            >
                <AnyReactComponent
                    lat={38.749447369341944}
                    lng={34.915717273488404}
                    text="My Marker"
                />
            </GoogleMapReact>
        </div>
    );
}
