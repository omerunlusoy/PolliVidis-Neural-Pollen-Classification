import * as React from 'react';
import Switch from '@mui/material/Switch';
import {
    Box,
    Card,
    CardActionArea,
    CardContent,
    FormControlLabel,
    Menu,
    MenuItem,
    MenuList,
    Paper,
    Typography
} from "@material-ui/core";

export default function PollenTypes() {
    const [checked, setChecked] = React.useState(true);

    const handleChange = (event) => {
        setChecked(event.target.checked);
    };

    const names = [
        'Ambrosia',
        'Alnus',
        'Acer',
        'Betula',
        'Juglans',
        'Artemisia',
        'Populus',
        'Phleum',
        'Pinaceae',
        'Junipenis',
        'Ulmus',
        'Quercus',
        'Carpinus',
        'Ligustrum',
        'Rumex',
        'Ailantus',
    ];

    return (
                    <Paper sx={{ width: 320, maxWidth: '100%' }}>
                        <MenuList
                        open={true}
                         >
                        {names.map((name) => (
                            <MenuItem
                                key={name}
                                value={name}
                            >
                                <FormControlLabel
                                    value="end"
                                    control={<Switch color="warning"
                                                     checked={checked}
                                                     onChange={handleChange} />}
                                    label={name}
                                    labelPlacement="end"
                                />
                            </MenuItem>
                        ))}

                        </MenuList>
                    </Paper>
    );
}
