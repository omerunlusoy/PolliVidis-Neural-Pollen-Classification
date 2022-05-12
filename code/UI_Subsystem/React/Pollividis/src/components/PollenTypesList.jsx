import * as React from 'react';
import Switch from '@mui/material/Switch';
import {
    Box, Button,
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
import { alpha, styled } from '@mui/material/styles';
import axios from "axios";

const RedSwitch = styled(Switch)(({ theme }) => ({
    '& .MuiSwitch-switchBase.Mui-checked': {
        color: '#A6232A',
        '&:hover': {
            backgroundColor: alpha('#A6232A', theme.palette.action.hoverOpacity),
        },
    },
    '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
        backgroundColor: '#A6232A',
    },
}));

export default function PollenTypes(props) {

    const [pollenArr,setPollenArr] = React.useState(props.pollens);
    const [checkedArr,setCheckedArr] = React.useState(props.checkedArr);

    const [checked1, setChecked1] = React.useState(checkedArr[0]);
    const [checked2, setChecked2] = React.useState(checkedArr[1]);
    const [checked3, setChecked3] = React.useState(checkedArr[2]);
    const [checked4, setChecked4] = React.useState(checkedArr[3]);
    const [checked5, setChecked5] = React.useState(checkedArr[4]);
    const [checked6, setChecked6] = React.useState(checkedArr[5]);
    const [checked7, setChecked7] = React.useState(checkedArr[6]);
    const [checked8, setChecked8] = React.useState(checkedArr[7]);
    const [checked9, setChecked9] = React.useState(checkedArr[8]);
    const [checked10, setChecked10] = React.useState(checkedArr[9]);
    const [checked11, setChecked11] = React.useState(checkedArr[10]);
    const [checked12, setChecked12] = React.useState(checkedArr[11]);
    const [checked13, setChecked13] = React.useState(checkedArr[12]);
    const [checked14, setChecked14] = React.useState(checkedArr[13]);
    const [checked15, setChecked15] = React.useState(checkedArr[14]);
    const [checked16, setChecked16] = React.useState(checkedArr[15]);
    const [checked17, setChecked17] = React.useState(checkedArr[16]);
    const [checked18, setChecked18] = React.useState(checkedArr[17]);
    const [checked19, setChecked19] = React.useState(checkedArr[18]);
    const [checked20, setChecked20] = React.useState(checkedArr[19]);
    const [checked21, setChecked21] = React.useState(checkedArr[20]);
    const [checked22, setChecked22] = React.useState(checkedArr[21]);
    const [checked23, setChecked23] = React.useState(checkedArr[22]);

    const [markers, setMarkers] = React.useState(props.markers);


    const heyu = [
        'ambrosia_artemisiifolia',
        'alnus_glutinosa',
        'acer_negundo',
        'betula_papyrifera',
        'juglans_regia',
        'artemisia_vulgaris',
        'populus_nigra',
        'phleum_phleoides',
        'picea_abies',
        'juniperus_communis',
        'ulmus_minor',
        'quercus_robur',
        'carpinus_betulus',
        'ligustrum_robustrum',
        'rumex_stenophyllus',
        'ailanthus_altissima',
        'thymbra_spicata',
        'rubia_peregrina',
        'olea_europaea',
        'cichorium_intybus',
        'chenopodium_album',
        'borago_officinalis',
        'acacia_dealbata'
    ];




    const handleChange1 = (event) => {
        setChecked1(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'ambrosia_artemisiifolia');

            setPollenArr(newList);
        }
    };

    const handleChange2 = (event) => {
        setChecked2(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'alnus_glutinosa');

            setPollenArr(newList);
        }
    };

    const handleChange3 = (event) => {
        setChecked3(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'acer_negundo');

            setPollenArr(newList);
        }
    };

    const handleChange4 = (event) => {
        setChecked4(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'betula_papyrifera');

            setPollenArr(newList);
        }
    };


    const handleChange5 = (event) => {
        setChecked5(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'juglans_regia');

            setPollenArr(newList);
        }
    };

    const handleChange6 = (event) => {
        setChecked6(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'artemisia_vulgaris');

            setPollenArr(newList);
        }
    };

    const handleChange7 = (event) => {
        setChecked7(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'populus_nigra');

            setPollenArr(newList);
        }
    };

    const handleChange8 = (event) => {
        setChecked8(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'phleum_phleoides');

            setPollenArr(newList);
        }
    };

    const handleChange9 = (event) => {
        setChecked9(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'picea_abies');

            setPollenArr(newList);
        }
    };

    const handleChange10 = (event) => {
        setChecked10(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'juniperus_communis');

            setPollenArr(newList);
        }
    };

    const handleChange11 = (event) => {
        setChecked11(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'ulmus_minor');

            setPollenArr(newList);
        }
    };

    const handleChange12 = (event) => {
        setChecked12(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'quercus_robur');

            setPollenArr(newList);
        }
    };

    const handleChange13 = (event) => {
        setChecked13(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'carpinus_betulus');

            setPollenArr(newList);
        }
    };

    const handleChange14 = (event) => {
        setChecked14(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'ligustrum_robustrum');

            setPollenArr(newList);
        }
    };

    const handleChange15 = (event) => {
        setChecked15(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'rumex_stenophyllus');

            setPollenArr(newList);
        }
    };

    const handleChange16 = (event) => {
        setChecked16(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'ailanthus_altissima');

            setPollenArr(newList);
        }
    };

    const handleChange17 = (event) => {
        setChecked17(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'thymbra_spicata');

            setPollenArr(newList);
        }
    };

    const handleChange18 = (event) => {
        setChecked18(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'rubia_peregrina');

            setPollenArr(newList);
        }
    };

    const handleChange19 = (event) => {
        setChecked19(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'olea_europaea');

            setPollenArr(newList);
        }
    };

    const handleChange20 = (event) => {
        setChecked20(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'cichorium_intybus');

            setPollenArr(newList);
        }
    };

    const handleChange21 = (event) => {
        setChecked21(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'chenopodium_album');

            setPollenArr(newList);
        }
    };

    const handleChange22 = (event) => {
        setChecked22(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'borago_officinalis');

            setPollenArr(newList);
        }
    };

    const handleChange23 = (event) => {
        setChecked23(event.target.checked);
        //console.log(event.target.value);
        if(event.target.checked){
            setPollenArr(pollenArr.concat([event.target.value]));
        }
        else{
            const newList = pollenArr.filter((item) => item !== 'acacia_dealbata');

            setPollenArr(newList);
        }
    };


    //console.log(pollenArr)

    const handleOk = () => {

        checkedArr[0] = checked1;
        checkedArr[1] = checked2;
        checkedArr[2] = checked3;
        checkedArr[3] = checked4;
        checkedArr[4] = checked5;
        checkedArr[5] = checked6;
        checkedArr[6] = checked7;
        checkedArr[7] = checked8;
        checkedArr[8] = checked9;
        checkedArr[9] = checked10;
        checkedArr[10] = checked11;
        checkedArr[11] = checked12;
        checkedArr[12] = checked13;
        checkedArr[13] = checked14;
        checkedArr[14] = checked15;
        checkedArr[15] = checked16;
        checkedArr[16] = checked17;
        checkedArr[17] = checked18;
        checkedArr[18] = checked19;
        checkedArr[19] = checked20;
        checkedArr[20] = checked21;
        checkedArr[21] = checked22;
        checkedArr[22] = checked23;


        let callBackArr = [pollenArr,checkedArr,false,markers]
        props.parentCallback(callBackArr);

        console.log("child :", callBackArr)


    };

    const handleSelectAll = () => {

        setChecked1(true);
        setChecked2(true);
        setChecked3(true);
        setChecked4(true);
        setChecked5(true);
        setChecked6(true);
        setChecked7(true);
        setChecked8(true);
        setChecked9(true);
        setChecked10(true);
        setChecked11(true);
        setChecked12(true);
        setChecked13(true);
        setChecked14(true);
        setChecked15(true);
        setChecked16(true);
        setChecked17(true);
        setChecked18(true);
        setChecked19(true);
        setChecked20(true);
        setChecked21(true);
        setChecked22(true);
        setChecked23(true);

        setPollenArr(heyu);

    };

    const handleUnSelectAll = () => {

        setChecked1(false);
        setChecked2(false);
        setChecked3(false);
        setChecked4(false);
        setChecked5(false);
        setChecked6(false);
        setChecked7(false);
        setChecked8(false);
        setChecked9(false);
        setChecked10(false);
        setChecked11(false);
        setChecked12(false);
        setChecked13(false);
        setChecked14(false);
        setChecked15(false);
        setChecked16(false);
        setChecked17(false);
        setChecked18(false);
        setChecked19(false);
        setChecked20(false);
        setChecked21(false);
        setChecked22(false);
        setChecked23(false);

        setPollenArr([]);

    };



    return (
                    <Paper sx={{ width: 600, maxWidth: '100%' }}>
                        <FormControlLabel
                            value="1"
                            control={<RedSwitch color="warning"
                                             checked={checked1}
                                             value="ambrosia_artemisiifolia"
                                             onChange={handleChange1} />}
                            label={"Ambrosia Artemisiifolia"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="2"
                            control={<RedSwitch color="warning"
                                             checked={checked2}
                                             value={"alnus_glutinosa"}
                                             onChange={handleChange2} />}
                            label={"Alnus Glutinosa"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="3"
                            control={<RedSwitch color="warning"
                                             checked={checked3}
                                             value="acer_negundo"
                                             onChange={handleChange3} />}
                            label={"Acer Negundo"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="4"
                            control={<RedSwitch color="warning"
                                             checked={checked4}
                                             value={"betula_papyrifera"}
                                             onChange={handleChange4} />}
                            label={"Betula Papyrifera"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="5"
                            control={<RedSwitch color="warning"
                                             checked={checked5}
                                             value="juglans_regia"
                                             onChange={handleChange5} />}
                            label={"Juglans Regia"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="6"
                            control={<RedSwitch color="warning"
                                             checked={checked6}
                                             value={"artemisia_vulgaris"}
                                             onChange={handleChange6} />}
                            label={"Artemisia Vulgaris"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="7"
                            control={<RedSwitch color="warning"
                                             checked={checked7}
                                             value="populus_nigra"
                                             onChange={handleChange7} />}
                            label={"Populus Nigra"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="8"
                            control={<RedSwitch color="warning"
                                             checked={checked8}
                                             value={"phleum_phleoides"}
                                             onChange={handleChange8} />}
                            label={"Phleum Phleoides"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="9"
                            control={<RedSwitch color="warning"
                                             checked={checked9}
                                             value="picea_abies"
                                             onChange={handleChange9} />}
                            label={"Picea Abies"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="10"
                            control={<RedSwitch color="warning"
                                             checked={checked10}
                                             value={"juniperus_communis"}
                                             onChange={handleChange10} />}
                            label={"Juniperus Communis"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="11"
                            control={<RedSwitch color="warning"
                                             checked={checked11}
                                             value="ulmus_minor"
                                             onChange={handleChange11} />}
                            label={"Ulmus Minor"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="12"
                            control={<RedSwitch color="warning"
                                             checked={checked12}
                                             value={"quercus_robur"}
                                             onChange={handleChange12} />}
                            label={"Quercus Robur"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="13"
                            control={<RedSwitch color="warning"
                                             checked={checked13}
                                             value="carpinus_betulus"
                                             onChange={handleChange13} />}
                            label={"Carpinus Betulus"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="14"
                            control={<RedSwitch color="warning"
                                             checked={checked14}
                                             value={"ligustrum_robustrum"}
                                             onChange={handleChange14} />}
                            label={"Ligustrum Robustrum"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="15"
                            control={<RedSwitch color="warning"
                                             checked={checked15}
                                             value="rumex_stenophyllus"
                                             onChange={handleChange15} />}
                            label={"Rumex Stenophyllus"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="16"
                            control={<RedSwitch color="warning"
                                             checked={checked16}
                                             value={"ailanthus_altissima"}
                                             onChange={handleChange16} />}
                            label={"Ailanthus Altissima"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="17"
                            control={<RedSwitch color="warning"
                                             checked={checked17}
                                             value="thymbra_spicata"
                                             onChange={handleChange17} />}
                            label={"Thymbra Spicata"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="18"
                            control={<RedSwitch color="warning"
                                             checked={checked18}
                                             value={"rubia_peregrina"}
                                             onChange={handleChange18} />}
                            label={"Rubia Peregrina"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="19"
                            control={<RedSwitch color="warning"
                                             checked={checked19}
                                             value="olea_europaea"
                                             onChange={handleChange19} />}
                            label={"Olea Europaea"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="20"
                            control={<RedSwitch color="warning"
                                             checked={checked20}
                                             value={"cichorium_intybus"}
                                             onChange={handleChange20} />}
                            label={"Cichorium Intybus"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="21"
                            control={<RedSwitch color="warning"
                                             checked={checked21}
                                             value="chenopodium_album"
                                             onChange={handleChange21} />}
                            label={"Chenopodium Album"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="22"
                            control={<RedSwitch color="warning"
                                             checked={checked22}
                                             value={"borago_officinalis"}
                                             onChange={handleChange22} />}
                            label={"Borago Officinalis"}
                            labelPlacement="end"
                        />
                        <FormControlLabel
                            value="23"
                            control={<RedSwitch color="warning"
                                             checked={checked23}
                                             value="acacia_dealbata"
                                             onChange={handleChange23} />}
                            label={"Acacia Dealbata"}
                            labelPlacement="end"
                        />
                        <div align={"center"} style={{marginBottom: 10}}>
                            <Button onClick={handleSelectAll} variant="contained" style={{backgroundColor:'#A6232A', color:'white',marginRight:4}} size="medium" >
                                Select All
                            </Button>
                            <Button onClick={handleUnSelectAll} variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium" >
                                Unselect All
                            </Button>
                        </div>
                        <div align={"center"}>
                            <Button onClick={handleOk} variant="contained" style={{backgroundColor:'#A6232A', color:'white'}} size="medium" >
                                Ok
                            </Button>
                        </div>

                    </Paper>
    );
}
